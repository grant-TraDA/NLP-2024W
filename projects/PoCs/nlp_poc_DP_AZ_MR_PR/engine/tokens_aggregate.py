import spacy
from torch import Tensor


class TokenAggregate:
    def __init__(
        self,
        spacy_token: list[str],
        clean_model_tokens: list[str],
        dirty_model_tokens: list[str],
        NERs: list[str],
        model_exp: list[float],
    ) -> None:
        self.spacy_token = spacy_token
        self.clean_model_tokens = clean_model_tokens
        self.dirty_model_tokens = dirty_model_tokens
        self.NERs = NERs
        self.model_exp = model_exp

    def __str__(self):
        return (
            f"Is spacy NER: {self.NERs}\n"
            f"spacy token: {self.spacy_token}\n"
            f"Our model clean: {self.clean_model_tokens}\n"
            f"Our model dirty: {self.dirty_model_tokens}\n"
            f"model exp: {self.model_exp}\n"
        )
    
    def get_tokens(self, get_clear:bool = False) -> list[str]:
        if get_clear:
            return self.clean_model_tokens
        else:
            return self.dirty_model_tokens

    @staticmethod
    def generate_aggregate_list(
        doc: spacy.tokens.doc.Doc,
        exp_tensor: Tensor,
        tokens_clear: list[str],
        tokens_dirty: list[str],
    ):

        spacy_token_to_our_tokens = []

        spacy_tokens = []
        NER = []

        current_clear_token_id = 0
        current_spacy_token_id = 0

        constructed_model_token = ""
        constructed_spacy_token = ""
        constructed_NERs: list[str] = []
        spacy_token_to_current_model_tokens: list[str] = []
        clean_tokens_for_current_spacy_tokens: list[str] = []
        dirty_tokens_for_current_spacy_tokens: list[str] = []
        model_token_exp = []

        for spacy_token in doc:
            spacy_tokens.append(spacy_token.text)
            NER.append(spacy_token.ent_type_)

        # xdd
        if spacy_tokens[0] == " ":
            current_spacy_token_id = 1

        while current_clear_token_id < len(
            tokens_clear
        ) or current_spacy_token_id < len(spacy_tokens):
            if (
                current_clear_token_id > len(tokens_clear)
            ) or current_spacy_token_id > len(spacy_tokens):
                break

            if len(constructed_model_token) < len(constructed_spacy_token):

                constructed_model_token += tokens_clear[current_clear_token_id]
                clean_tokens_for_current_spacy_tokens.append(
                    tokens_clear[current_clear_token_id]
                )
                dirty_tokens_for_current_spacy_tokens.append(
                    tokens_dirty[current_clear_token_id + 1]
                )
                model_token_exp.append(
                    float(exp_tensor[0, current_clear_token_id + 1])
                )
                current_clear_token_id += 1
            else:
                if current_spacy_token_id == len(spacy_tokens):
                    break
                constructed_spacy_token += spacy_tokens[current_spacy_token_id]
                constructed_NERs.append(NER[current_spacy_token_id])
                spacy_token_to_current_model_tokens.append(
                    spacy_tokens[current_spacy_token_id]
                )
                current_spacy_token_id += 1

            if constructed_model_token == constructed_spacy_token:

                new_aggregate = TokenAggregate(
                    spacy_token_to_current_model_tokens,
                    clean_tokens_for_current_spacy_tokens,
                    dirty_tokens_for_current_spacy_tokens,
                    constructed_NERs,
                    model_token_exp,
                )
                spacy_token_to_our_tokens.append(new_aggregate)

                constructed_model_token = ""
                constructed_spacy_token = ""
                constructed_NERs = []
                spacy_token_to_current_model_tokens = []
                clean_tokens_for_current_spacy_tokens = []
                dirty_tokens_for_current_spacy_tokens = []
                model_token_exp = []

        # debug!!!
        if len(constructed_model_token):
            print(f"\n INVALID DOC!!! stopped")
            print(constructed_spacy_token)
            print(constructed_model_token)
            return False

        return spacy_token_to_our_tokens
