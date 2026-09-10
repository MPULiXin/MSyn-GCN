from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F


def _to_torch_sparse(matrix: sparse.csr_matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def _l2_normalize_rows(values: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return values / values.norm(p=2, dim=1, keepdim=True).clamp_min(eps)


def _normalize_rows(values: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return values / values.sum(dim=1, keepdim=True).clamp_min(eps)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Projection onto the probability simplex (Martins and Astudillo, 2016)."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    ranks = torch.arange(
        1,
        logits.size(dim) + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    shape = [1] * logits.dim()
    shape[dim] = -1
    ranks = ranks.view(shape)
    cumulative = sorted_logits.cumsum(dim=dim)
    support = (1 + ranks * sorted_logits) > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(dim, support_size.to(torch.long) - 1) - 1
    ) / support_size.to(logits.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


@dataclass
class ModelOutput:
    herb_logits: torch.Tensor
    herb_probabilities: torch.Tensor
    syndrome_embedding: torch.Tensor
    herb_embeddings: torch.Tensor
    eight_distribution: torch.Tensor
    zangfu_distribution: torch.Tensor
    eight_logits: torch.Tensor | None
    zangfu_logits: torch.Tensor | None
    symptom_attention: torch.Tensor | None
    syndrome_gate: torch.Tensor | None


class MSynGCN(nn.Module):
    """Paper-aligned Multi-Syndrome Graph Convolutional Network."""

    def __init__(
        self,
        *,
        n_symptoms: int,
        n_herbs: int,
        bipartite_adj: sparse.csr_matrix,
        symptom_adj: sparse.csr_matrix,
        herb_adj: sparse.csr_matrix,
        properties: dict[str, np.ndarray],
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        rule_matrices: dict[str, list[list[float]]],
    ) -> None:
        super().__init__()
        self.n_symptoms = int(n_symptoms)
        self.n_herbs = int(n_herbs)
        self.model_config = dict(model_config)
        self.training_config = dict(training_config)

        embedding_size = int(model_config["embedding_size"])
        layer_sizes = [int(value) for value in model_config["graph_layer_sizes"]]
        if not layer_sizes:
            raise ValueError("graph_layer_sizes must not be empty")
        self.graph_fusion = str(model_config["graph_fusion"])
        self.attention_pooling = model_config["symptom_pooling"] == "attention"
        self.property_fusion = str(model_config["property_fusion"])
        self.use_property_representation = bool(model_config["use_property_representation"])
        self.diagnostic_enabled = bool(model_config["diagnostic_enabled"])
        self.syndrome_activation = str(model_config["syndrome_activation"])
        self.dropout_probability = float(model_config["dropout"])

        self.lambda_align = float(training_config["lambda_align"])
        self.lambda_qi = float(training_config["lambda_qi"])
        self.lambda_flavor = float(training_config["lambda_flavor"])
        self.lambda_meridian = float(training_config["lambda_meridian"])
        self.lambda_marginal = float(training_config["lambda_marginal"])
        self.lambda_syndrome = float(training_config["lambda_syndrome"])
        self.lambda_regularization = float(
            training_config["lambda_representation_regularization"]
        )

        self.register_buffer("bipartite_adj", _to_torch_sparse(bipartite_adj))
        self.register_buffer("symptom_adj", _to_torch_sparse(symptom_adj))
        self.register_buffer("herb_adj", _to_torch_sparse(herb_adj))

        self.symptom_embedding = nn.Parameter(torch.empty(n_symptoms, embedding_size))
        self.herb_embedding = nn.Parameter(torch.empty(n_herbs, embedding_size))
        nn.init.xavier_uniform_(self.symptom_embedding)
        nn.init.xavier_uniform_(self.herb_embedding)

        dimensions = [embedding_size, *layer_sizes]
        self.message_symptom = nn.ModuleList()
        self.message_herb = nn.ModuleList()
        self.update_symptom = nn.ModuleList()
        self.update_herb = nn.ModuleList()
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
            self.message_symptom.append(nn.Linear(input_size, input_size, bias=False))
            self.message_herb.append(nn.Linear(input_size, input_size, bias=False))
            self.update_symptom.append(nn.Linear(input_size * 2, output_size))
            self.update_herb.append(nn.Linear(input_size * 2, output_size))

        last_graph_size = layer_sizes[-1]
        self.same_type_symptom = nn.Linear(embedding_size, last_graph_size, bias=False)
        self.same_type_herb = nn.Linear(embedding_size, last_graph_size, bias=False)
        if self.graph_fusion == "concat":
            self.model_size = last_graph_size * 2
        elif self.graph_fusion == "add":
            self.model_size = last_graph_size
        else:
            raise ValueError("graph_fusion must be 'concat' or 'add'")

        mlp_sizes = [int(value) for value in model_config["mlp_sizes"]]
        if not mlp_sizes or mlp_sizes[-1] != self.model_size:
            raise ValueError(
                f"mlp_sizes must end at the fused graph size {self.model_size}; got {mlp_sizes}"
            )
        mlp_layers: list[nn.Module] = []
        previous = self.model_size
        for index, size in enumerate(mlp_sizes):
            mlp_layers.append(nn.Linear(previous, size))
            if index < len(mlp_sizes) - 1:
                mlp_layers.extend([nn.ReLU(), nn.Dropout(self.dropout_probability)])
            previous = size
        self.symptom_projection = nn.Sequential(*mlp_layers)
        self.pre_mlp_dropout = nn.Dropout(self.dropout_probability)
        if self.attention_pooling:
            self.attention_scorer = nn.Linear(self.model_size, 1)

        qi = torch.as_tensor(properties["qi"], dtype=torch.float32)
        flavor = torch.as_tensor(properties["flavor"], dtype=torch.float32)
        meridian = torch.as_tensor(properties["meridian"], dtype=torch.float32)
        self.register_buffer("property_qi", qi)
        self.register_buffer("property_flavor", flavor)
        self.register_buffer("property_meridian", meridian)
        self.register_buffer("target_qi", _normalize_rows(qi))
        self.register_buffer("target_flavor", _normalize_rows(flavor))
        self.register_buffer("target_meridian", _normalize_rows(meridian))

        property_channel_size = last_graph_size
        self.project_qi = nn.Linear(qi.shape[1], property_channel_size, bias=False)
        self.project_flavor = nn.Linear(flavor.shape[1], property_channel_size, bias=False)
        self.project_meridian = nn.Linear(meridian.shape[1], property_channel_size, bias=False)
        self.property_projection = nn.Linear(property_channel_size * 3, self.model_size)
        if self.property_fusion == "gate":
            self.herb_gate = nn.Sequential(
                nn.Linear(self.model_size * 2, self.model_size),
                nn.Sigmoid(),
            )
        elif self.property_fusion == "concat":
            self.herb_concat = nn.Linear(self.model_size * 2, self.model_size)
        elif self.property_fusion != "add":
            raise ValueError("property_fusion must be 'gate', 'concat', or 'add'")

        if self.diagnostic_enabled:
            head_size = int(model_config["diagnostic_head_hidden_size"])
            self.eight_head = nn.Sequential(
                nn.Linear(self.model_size, head_size),
                nn.ReLU(),
                nn.Linear(head_size, 8),
            )
            self.zangfu_head = nn.Sequential(
                nn.Linear(self.model_size, head_size),
                nn.ReLU(),
                nn.Linear(head_size, 12),
            )
            self.eight_prototypes = nn.Parameter(torch.empty(8, head_size))
            self.zangfu_prototypes = nn.Parameter(torch.empty(12, head_size))
            nn.init.xavier_uniform_(self.eight_prototypes)
            nn.init.xavier_uniform_(self.zangfu_prototypes)
            self.cross_knowledge_gate = nn.Sequential(
                nn.Linear(head_size * 2, head_size),
                nn.ReLU(),
                nn.Linear(head_size, 2),
                nn.Softmax(dim=-1),
            )
            self.knowledge_projection = nn.Linear(head_size * 2, self.model_size)
            self.syndrome_gate = nn.Sequential(
                nn.Linear(self.model_size * 2, self.model_size),
                nn.Sigmoid(),
            )

        self.register_buffer(
            "eight_to_qi",
            torch.tensor(rule_matrices["eight_to_qi"], dtype=torch.float32),
        )
        self.register_buffer(
            "eight_to_flavor",
            torch.tensor(rule_matrices["eight_to_flavor"], dtype=torch.float32),
        )
        self.register_buffer("zangfu_to_meridian", torch.eye(12, dtype=torch.float32))

    def _encode_graphs(self) -> tuple[torch.Tensor, torch.Tensor]:
        symptoms = self.symptom_embedding
        herbs = self.herb_embedding
        initial_symptoms = symptoms
        initial_herbs = herbs
        for symptom_message, herb_message, symptom_update, herb_update in zip(
            self.message_symptom,
            self.message_herb,
            self.update_symptom,
            self.update_herb,
        ):
            combined = torch.cat([symptoms, herbs], dim=0)
            messages = torch.sparse.mm(self.bipartite_adj, combined)
            symptom_messages = symptom_message(messages[: self.n_symptoms])
            herb_messages = herb_message(messages[self.n_symptoms :])
            symptoms = torch.tanh(symptom_update(torch.cat([symptoms, symptom_messages], dim=1)))
            herbs = torch.tanh(herb_update(torch.cat([herbs, herb_messages], dim=1)))
            symptoms = _l2_normalize_rows(
                F.dropout(symptoms, p=self.dropout_probability, training=self.training)
            )
            herbs = _l2_normalize_rows(
                F.dropout(herbs, p=self.dropout_probability, training=self.training)
            )

        same_symptoms = torch.tanh(
            self.same_type_symptom(torch.sparse.mm(self.symptom_adj, initial_symptoms))
        )
        same_herbs = torch.tanh(
            self.same_type_herb(torch.sparse.mm(self.herb_adj, initial_herbs))
        )
        if self.graph_fusion == "concat":
            return torch.cat([symptoms, same_symptoms], dim=1), torch.cat([herbs, same_herbs], dim=1)
        return symptoms + same_symptoms, herbs + same_herbs

    def _herb_representation(self, structural: torch.Tensor) -> torch.Tensor:
        if not self.use_property_representation:
            return structural
        property_embedding = self.property_projection(
            torch.cat(
                [
                    self.project_qi(self.property_qi),
                    self.project_flavor(self.property_flavor),
                    self.project_meridian(self.property_meridian),
                ],
                dim=1,
            )
        )
        if self.property_fusion == "gate":
            gate = self.herb_gate(torch.cat([structural, property_embedding], dim=1))
            return gate * property_embedding + (1.0 - gate) * structural
        if self.property_fusion == "concat":
            return self.herb_concat(torch.cat([structural, property_embedding], dim=1))
        return structural + property_embedding

    def _pool_symptoms(
        self,
        symptom_multi_hot: torch.Tensor,
        symptom_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.attention_pooling:
            observed = symptom_multi_hot.bool()
            if not observed.any(dim=1).all():
                raise ValueError("Every prescription must contain at least one symptom")
            scores = self.attention_scorer(symptom_embeddings).squeeze(1)
            masked = scores.unsqueeze(0).expand_as(symptom_multi_hot).masked_fill(~observed, -torch.inf)
            attention = torch.softmax(masked, dim=1)
            return attention @ symptom_embeddings, attention
        counts = symptom_multi_hot.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (symptom_multi_hot @ symptom_embeddings) / counts, None

    def _activate_syndromes(self, logits: torch.Tensor) -> torch.Tensor:
        if self.syndrome_activation == "sparsemax":
            return sparsemax(logits, dim=1)
        if self.syndrome_activation == "softmax":
            return torch.softmax(logits, dim=1)
        raise ValueError(f"Unknown syndrome activation: {self.syndrome_activation}")

    def forward(self, symptom_multi_hot: torch.Tensor) -> ModelOutput:
        symptom_embeddings, structural_herbs = self._encode_graphs()
        herb_embeddings = self._herb_representation(structural_herbs)
        pooled, attention = self._pool_symptoms(symptom_multi_hot, symptom_embeddings)
        data_driven = self.symptom_projection(self.pre_mlp_dropout(pooled))

        if self.diagnostic_enabled:
            eight_logits = self.eight_head(pooled)
            zangfu_logits = self.zangfu_head(pooled)
            eight = self._activate_syndromes(eight_logits)
            zangfu = self._activate_syndromes(zangfu_logits)
            eight_embedding = eight @ self.eight_prototypes
            zangfu_embedding = zangfu @ self.zangfu_prototypes
            cross_gate = self.cross_knowledge_gate(
                torch.cat([eight_embedding, zangfu_embedding], dim=1)
            )
            knowledge = self.knowledge_projection(
                torch.cat(
                    [
                        cross_gate[:, :1] * eight_embedding,
                        cross_gate[:, 1:] * zangfu_embedding,
                    ],
                    dim=1,
                )
            )
            gate = self.syndrome_gate(torch.cat([data_driven, knowledge], dim=1))
            syndrome = gate * knowledge + (1.0 - gate) * data_driven
        else:
            batch_size = symptom_multi_hot.shape[0]
            eight = symptom_multi_hot.new_zeros((batch_size, 8))
            zangfu = symptom_multi_hot.new_zeros((batch_size, 12))
            eight_logits = None
            zangfu_logits = None
            gate = None
            syndrome = data_driven

        herb_logits = syndrome @ herb_embeddings.transpose(0, 1)
        return ModelOutput(
            herb_logits=herb_logits,
            herb_probabilities=torch.sigmoid(herb_logits),
            syndrome_embedding=syndrome,
            herb_embeddings=herb_embeddings,
            eight_distribution=eight,
            zangfu_distribution=zangfu,
            eight_logits=eight_logits,
            zangfu_logits=zangfu_logits,
            symptom_attention=attention,
            syndrome_gate=gate,
        )

    @staticmethod
    def _symmetric_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        first = first.clamp_min(eps)
        second = second.clamp_min(eps)
        forward = (first * (first.log() - second.log())).sum(dim=1)
        reverse = (second * (second.log() - first.log())).sum(dim=1)
        return (forward + reverse).mean()

    def _alignment_loss(
        self,
        herbs: torch.Tensor,
        eight: torch.Tensor,
        zangfu: torch.Tensor,
    ) -> torch.Tensor:
        if not self.diagnostic_enabled or self.lambda_align == 0:
            return herbs.new_zeros(())
        predicted_qi = _normalize_rows(eight @ self.eight_to_qi)
        predicted_flavor = _normalize_rows(eight @ self.eight_to_flavor)
        predicted_meridian = _normalize_rows(zangfu @ self.zangfu_to_meridian)
        target_qi = _normalize_rows(herbs @ self.target_qi)
        target_flavor = _normalize_rows(herbs @ self.target_flavor)
        target_meridian = _normalize_rows(herbs @ self.target_meridian)
        return self.lambda_align * (
            self.lambda_qi * self._symmetric_kl(predicted_qi, target_qi)
            + self.lambda_flavor * self._symmetric_kl(predicted_flavor, target_flavor)
            + self.lambda_meridian * self._symmetric_kl(predicted_meridian, target_meridian)
        )

    def compute_loss(
        self,
        symptom_multi_hot: torch.Tensor,
        herb_multi_hot: torch.Tensor,
        item_weights: torch.Tensor,
        *,
        eight_labels: torch.Tensor | None,
        zangfu_labels: torch.Tensor | None,
        marginal_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], ModelOutput]:
        output = self(symptom_multi_hot)
        reconstruction = (
            (output.herb_probabilities - herb_multi_hot).square() * item_weights.unsqueeze(0)
        ).mean()
        alignment = self._alignment_loss(
            herb_multi_hot,
            output.eight_distribution,
            output.zangfu_distribution,
        )

        syndrome_loss = reconstruction.new_zeros(())
        if self.lambda_syndrome > 0:
            if eight_labels is None or zangfu_labels is None:
                raise ValueError("The paper variant requires training-split weak labels")
            if output.eight_logits is None or output.zangfu_logits is None:
                raise ValueError("Syndrome supervision requires the diagnostic module")
            valid = symptom_multi_hot.sum(dim=1) > 0
            per_row = -(
                eight_labels * F.log_softmax(output.eight_logits, dim=1)
            ).sum(dim=1) - (
                zangfu_labels * F.log_softmax(output.zangfu_logits, dim=1)
            ).sum(dim=1)
            syndrome_loss = self.lambda_syndrome * per_row[valid].mean()

        marginal_loss = reconstruction.new_zeros(())
        if self.lambda_marginal > 0:
            if marginal_targets is None:
                raise ValueError("The full paper variant requires train-only marginal targets")
            valid = marginal_targets.sum(dim=1) > 0
            per_row = -(
                marginal_targets * F.log_softmax(output.herb_logits, dim=1)
            ).sum(dim=1)
            marginal_loss = self.lambda_marginal * per_row[valid].mean()

        regularization = self.lambda_regularization * (
            output.syndrome_embedding.square().mean()
            + output.herb_embeddings.square().mean()
        )
        total = reconstruction + syndrome_loss + marginal_loss + alignment + regularization
        parts = {
            "total": total.detach(),
            "reconstruction": reconstruction.detach(),
            "syndrome": syndrome_loss.detach(),
            "marginal": marginal_loss.detach(),
            "alignment": alignment.detach(),
            "regularization": regularization.detach(),
        }
        return total, parts, output

