import torch
import warnings
from typing import (
    Iterable,
    Union,
    List,
    Collection,
    Optional,
    Callable,
    Tuple,
    Any,
    cast,
)
from transformers import (
    AutoModelForCausalLM,
)
from minicons.scorer import IncrementalLMScorer


class IncrementalBCEScorer(IncrementalLMScorer):
    def __init__(
            self,
            model: Union[str, torch.nn.Module],
            device: Optional[str] = "cpu",
            tokenizer=None,
            **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as a Huggingface model obtained from
            `AutoModelForCausalLM`. In the last case, a corresponding tokenizer
            must also be provided.
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        super(IncrementalBCEScorer, self).__init__(
            model, device=device, tokenizer=tokenizer
        )

        if isinstance(model, str):
            if self.device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, device_map=self.device, return_dict=True, **kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, return_dict=True, **kwargs
                )
        else:
            self.model = model

        if self.device != "auto":
            self.model.to(self.device)

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding pad_token_id to the tokenizer."
                )
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<pad>"]}
                )
                self.tokenizer.pad_token = "<pad>"
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

        if isinstance(model, str):
            self.model.eval()

        self.padding_side = self.tokenizer.padding_side

    def compute_stats(
            self,
            batch: Iterable,
            rank: bool = False,
            prob: bool = False,
            base_two: bool = False,
            return_tensors: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
                base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, offsets = batch
        if self.device != "auto":
            encoded = encoded.to(self.device)

        # ids = [
        #     [i for i in instance if i != self.tokenizer.pad_token_id]
        #     for instance in encoded["input_ids"].tolist()
        # ]
        ids = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
            )
        ]

        ## Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(**encoded).logits.detach()

        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            logit = logit.squeeze(0)[torch.arange(offset, length),]

            # this line computes log_probability using the log-sum-exp trick (i.e. no need for Softmax)
            # logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            # can I just directly replace with with log(sigmoid)? Or do we need a probability distribution here?
            logprob_distribution = torch.log(torch.sigmoid(logit))

            # print(logit.size(), logit[0])
            # print(logprob_distribution.size(), logprob_distribution[0])
            # print(new_logprob_distribution.size(), new_logprob_distribution[0])
            # exit()

            query_ids = idx[offset:]
            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                        / torch.tensor(2).log()
                ).tolist()
            else:
                if prob:
                    score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                            .exp()
                            .tolist()
                    )
                else:
                    score = logprob_distribution[
                        torch.arange(length - offset), query_ids
                    ].tolist()

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, ranks
        else:
            return scores
