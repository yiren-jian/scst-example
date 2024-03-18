import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

if __name__ == "__main__":
    device = torch.device("cuda:0")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    model = model.to(device)

    input_text = [
        "translate English to German: How old are you?",
        "translate Chinese to Korean: What is your name?",
    ]

    # output_text = [
    #     "I do not know English",
    #     "I do not know Chinese",
    # ]
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        input_tokens = tokenizer(
            input_text,
            padding="longest",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to(device)

        inputs_embeds = model.encoder.embed_tokens(input_tokens.input_ids)
        
        Bsz = inputs_embeds.shape[0]
        num_beams = 5

        outputs = model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=input_tokens.attention_mask,
            do_sample=False,
            top_p=0.9,
            temperature=1,
            num_beams=5,
            max_length=32,
            # min_length=5, ### RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # eos_token_id=self.eos_token_id, ### AttributeError
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_return_sequences=5, ### num_beams
            return_dict_in_generate=True,
            output_scores=True,
        )

        assert len(outputs.sequences) == Bsz * num_beams

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        )

        output_length = torch.sum(transition_scores < 0, dim=1)
        sequences_scores = transition_scores.sum(dim=1) / (output_length**1.0)
        sequences_scores = sequences_scores.view(Bsz, -1) # [batch, num_beams]

        reward = torch.randn(Bsz, 5, dtype=torch.float16).to(device) # fake a reward (should be our helpfulness score, etc.)
        reward_baseline = torch.mean(reward, -1, keepdim=True)
        loss = - (sequences_scores) * (reward-reward_baseline)
        loss = loss.mean()

        loss.backward()