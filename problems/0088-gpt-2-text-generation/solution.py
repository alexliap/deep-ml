def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	encoder, hparams, params = load_encoder_hparams_and_params()
	context_length = hparams["n_ctx"]
	enc_prompt = encoder.encode(prompt)[-context_length:]
	result = []
	for i in range(n_tokens_to_generate):
		L = len(enc_prompt)
		# embeddings
		x = params["wte"][enc_prompt, :] # L, C
		x_p_embped = params["wpe"][:L, :] # L, C
		x += x_p_embped # L, C
		# blocks
		for block in params["blocks"]:
			# layernorm 1
			mean = x.mean(axis=-1, keepdims=True)    # (L, 1)
			var = x.var(axis=-1, keepdims=True)      # (L, 1)
			ln_1 = block["ln_1"]["g"] * (x - mean) / np.sqrt(var + 1e-5) + block["ln_1"]["b"]

			wq, wk, wv = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
			bq, bk, bv = np.split(block["attn"]["c_attn"]["b"], 3, axis=-1)

			q, k, v = (ln_1 @ wq) + bq, (ln_1 @ wk) + bk, (ln_1 @ wv) + bv
			q = q.reshape(L, hparams["n_head"], -1).transpose(1,0,2) # (n_head, L, head_dim)
			k = k.reshape(L, hparams["n_head"], -1).transpose(1,0,2) # (n_head, L, head_dim)
			v = v.reshape(L, hparams["n_head"], -1).transpose(1,0,2) # (n_head, L, head_dim)

			scores = (q @ k.transpose(0,2,1))/np.sqrt(k.shape[-1]) # (n_head, L, L)
			# causal mask
			mask = np.zeros((hparams["n_head"], L, L))
			for j in range(scores.shape[-1]):
				mask[:, :j, j] = - np.inf
			scores += mask
			soft_nom = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
			softmax = soft_nom/np.sum(soft_nom, axis=-1, keepdims=True)

			attn = softmax @ v # (n_head, L, head_dim)
			attn = attn.transpose(1, 0, 2)
			attn = attn.reshape(L, -1)

			x = x + (attn @ block["attn"]["c_proj"]["w"]) + block["attn"]["c_proj"]["b"]

			# layernorm 2
			mean = x.mean(axis=-1, keepdims=True)    # (L, 1)
			var = x.var(axis=-1, keepdims=True)      # (L, 1)
			ln_2 = block["ln_2"]["g"] * (x - mean) / np.sqrt(var + 1e-5) + block["ln_2"]["b"]

			def gelu(x):
				return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
			# mlp
			x_mlp = (ln_2 @ block["mlp"]["c_fc"]["w"]) + block["mlp"]["c_fc"]["b"]
			x_mlp = gelu(x_mlp)
			x_mlp = (x_mlp @ block["mlp"]["c_proj"]["w"]) + block["mlp"]["c_proj"]["b"]
			x = x + x_mlp

		# layernorm 3
		mean = x.mean(axis=-1, keepdims=True)    # (L, 1)
		var = x.var(axis=-1, keepdims=True)      # (L, 1)
		x = params["ln_f"]["g"] * (x - mean) / np.sqrt(var + 1e-5) + params["ln_f"]["b"]

		x = x @ params["wte"].T

		logits = x[-1, :]

		chosen_id = int(np.argmax(logits))

		enc_prompt.append(chosen_id)
		enc_prompt = enc_prompt[-context_length:]

		result.append(chosen_id)
		result = result[-context_length:]

	return encoder.decode(result)




def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 2
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [
			{
				"mlp": {
					"c_fc": {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
					"c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
				},
				"attn": {
					"c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
					"c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
				},
				"ln_1": {"g": np.ones(10), "b": np.zeros(10)},
				"ln_2": {"g": np.ones(10), "b": np.zeros(10)},
			}
		],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params