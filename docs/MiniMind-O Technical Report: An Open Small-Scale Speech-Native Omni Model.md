# MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model

Jingyao Gong Independent Researcher gongjy.cs@foxmail.com 

github.com/jingyaogong/minimind-o 0 huggingface.co/collections/jingyaogong/minimind-o modelscope.cn/collections/gongjy/minimind-o 

## Abstract

MiniMind-O is an open 0.1B-scale omni model built on the MiniMind language model [Gong, 2024, 2026]. It accepts text, speech, and image inputs, and returns both text and streaming speech. The release includes model code, checkpoints, and the main Parquet training datasets for T2A, I2T, and A2A, making the complete interaction loop directly inspectable. The model uses a full MiniMind backbone as the Thinker and an independent four-layer Talker made from MiniMind blocks. Frozen SenseVoice-Small and SigLIP2 encoders provide speech and image features, which are mapped by lightweight MLP projectors and injected at modalityplaceholder positions. The Talker reads a middle-layer Thinker state together with an autoregressive eight-layer Mimi-code buffer. Speaker control is handled by a dedicated speaker token, right-aligned reference codec prompts, and precomputed 192-dimensional CAM++ embeddings, so voice conditioning remains part of the audio-code context rather than a separate TTS module. With a 768-dimensional Talker, the dense and MoE variants reach average CERs of 0.0897 and 0.0900 in Thinker–Talker consistency evaluation, with overall voice-cloning similarities of 0.5995 and 0.5937. Beyond reporting a working system, the paper identifies three scale-critical design choices for small omni models: middle-layer semantic bridging, a released multimodal sequence format, and a parameter-efficient eight-codebook interface. 

## 1 Introduction

Models such as GPT-4o, Qwen-Omni, Moshi, and recent speech-text systems have moved real-time multimodal interaction from a product interface problem into a model-design problem [Openai, 2024, Défossez et al., 2024, Xu et al., 2025a,b]. A usable system has to listen, see, reason, speak, and stop speaking when the user interrupts. The usual engineering path is still a cascade: ASR turns speech into text, an LLM writes the answer, and TTS renders the waveform. This path works, but it leaves the language model outside the acoustic loop. Once the speech module is external, errors in pronunciation, timing, and speaker control are hard to attribute to a shared representation. 

MiniMind-O takes the opposite constraint as the starting point. The base model is MiniMind, not a billion-scale backbone, so every added modality has to pass through a very small hidden space. This makes the system a useful stress test for omni-model design: components that are merely convenient at large scale have to become explicit and measurable at 0.1B scale. The design in Figure 1 keeps the semantic path and the acoustic path separate. The Thinker is the MiniMind transformer itself. It receives normal text embeddings, plus projected SenseVoice and SigLIP2 states injected at audio and image placeholder positions. The Talker is a separate four-layer module initialized from MiniMind blocks when compatible weights are available. This keeps semantic prediction in the language backbone and gives audio-code generation its own recurrent history. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/1c09e6e1f25b2a90421304a726312e43ef0811516d2fbf9d96248489ab408af8.jpg)



Figure 1: Architecture of MiniMind-O. Audio and image inputs are encoded by frozen SenseVoice and SigLIP2 encoders, mapped into the MiniMind hidden space by MLP projectors, and injected at modality-placeholder positions. A middle-layer Thinker state is fused with the Mimi-code history by an independent Talker, which predicts eight codec layers for streaming speech generation.


The size is not only a constraint; it is also the main experimental handle. MiniMind-O is intended as a small and fully inspectable omni implementation: it supports text, speech, and image inputs together with streaming speech output while keeping the active model around 0.1B parameters. At this scale, the bridge, projectors, and codec interface have to remain necessary, measurable, and reproducible. 

A second result is more architectural. The eight Mimi codebooks could have been given eight independent embedding tables and eight independent output heads. In practice, a non-full-rank shared-base-plus-adapter parameterization gives a clear parameter-efficiency curve: moderate ranks recover most of the convergence and codebook-accuracy gain, while the decoupled rank study shows that the output head rank matters more than the input embedding rank. This makes the low-rank interface an empirically supported design choice rather than an implementation shortcut. 

The bridge layer is the third point. If the Talker reads the final next-token-prediction state, it inherits a strong bias toward the current text token and the geometry of the LM head; this is useful for text logits but noisy as an acoustic condition. If it reads too shallow a state, the model has not yet accumulated enough context to resolve pronunciation, syntax, or cross-modal reference. A simple Mandarin example is the character U+5730, whose pronunciation can be de or di depending on context. A raw embedding does not encode this context-specific pronunciation, while a middle hidden state can carry enough surrounding information without being fully collapsed into the next-token classifier. 

The fourth part of the release is the dataset itself. Omni systems are hard to reproduce if the code is open but the alignment data, codec targets, and modality layout are implicit. MiniMind-O therefore releases the main T2A, I2T, and A2A Parquet datasets together with the code path that consumes them. The dataset is not meant to be a final universal corpus; it is the training substrate for this small-model recipe, with text, image bytes, speech inputs, Mimi code targets, reference-code prompts, and speaker embeddings organized in a format that can be inspected and modified. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/0f55c5695a19b381ee5e795c1ac326c691dad178865cc681dd11bd4d95845857.jpg)



Figure 2: Talker-side speech generation design. The Talker consumes the Thinker bridge state, audio-code embeddings, optional speaker information, and reference codec prompts, then emits eight-layer Mimi codebook logits for waveform decoding.


The released system has two variants: a dense minimind-3o model and a minimind-3o-moe model with roughly the same active scale. Audio input is encoded by SenseVoice-Small [An et al., 2024]; image input is encoded by SigLIP2 [Tschannen et al., 2025]; speech output is represented by eight Mimi codebooks and decoded to 24 kHz audio [Défossez et al., 2024]. Speaker conditioning is injected by two scale-compatible signals: reference codec prompts and 192-dimensional CAM++ speaker embeddings [Wang et al., 2023b]. This choice also keeps the inference path inspectable, because no speaker encoder is called inside the model forward pass. 

The voice path is therefore closer to in-context conditioning than to a fixed-speaker TTS head. The default release ships five built-in voice prompts, dylan, eric, serena, uncle_fu, and vivian; an additional seven voices are kept as held-out prompts for evaluation. At inference time, changing the voice only changes the right-aligned reference Mimi codes and the CAM++ vector placed at the <|audio_spk|> position. The Thinker prompt and Talker weights remain unchanged, which makes voice transfer a property of the shared audio-code layout rather than a separate fine-tuning path. 

The report documents the design factors identified as important in this small regime: where to extract the Thinker state, how wide the Talker has to be, how reference speech should be placed in the audio buffer, how the released data is organized, and which evaluation exposes content mismatch rather than only audio quality. These details are not incidental implementation choices. At 0.1B scale, bridge placement, reusable data, and parameter-efficient codebook interfaces directly affect whether the complete loop remains trainable and reproducible. The contribution is therefore not a new large model, but a compact and inspectable recipe that turns speech-native omni interaction into a controllable research object. 

## 2 Related Work

Omni and speech-text dialogue models. GPT-4o made speech-native multimodal interaction widely visible, and Qwen2.5-Omni and Qwen3-Omni later made the Thinker–Talker recipe more concrete: hidden states can be extracted from a semantic path and consumed by a speech path that runs in streaming mode [Openai, 2024, Xu et al., 2025a,b]. Open systems explore nearby choices. Mini-Omni showed that speech can be streamed while a language model is still generating text [Xie and Wu, 2024a]; Mini-Omni2 added vision and duplex interaction [Xie and Wu, 2024b]. LLaMA-Omni, VITA, GLM-4-Voice, Baichuan-Audio, Step-Audio, and Spirit-LM study related mixtures of speech interaction, audio-language understanding, and interleaved spoken-written modeling [Fang et al., 2024, Fu et al., 2024, Zeng et al., 2024, Li et al., 2025, Huang et al., 2025, Nguyen et al., 2025]. MiniMind-O uses this line of work as the reference point and studies a complementary question: which components remain necessary when the active model is pushed down to roughly 0.1B parameters, and which interface choices make the resulting loop reproducible rather than only demonstrable. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/0b273c591b35bb8b6092d7f7fc1a4a987f62dce44e5629075a9e9870055a0994.jpg)



Figure 3: Training sequence format for Thinker and Talker. Text supervision is applied to the Thinker response tokens, while audio supervision is applied to target Mimi code positions. Reference-code regions are used as conditioning context rather than loss targets.


Discrete audio representation and speech generation. Discrete audio tokens are the reason the Talker can be trained with a language-model-style objective. VALL-E showed that codec tokens can carry enough information for zero-shot TTS, MusicGen made multi-codebook autoregression a standard generation pattern, and EnCodec and SNAC provided practical neural codec choices [Wang et al., 2023a, Copet et al., 2024, Défossez et al., 2022, Siuzdak, 2024]. Moshi introduced Mimi as a streaming audio codec in a speech-text system, while MOSS-Audio-Tokenizer studies scalable tokenizer design for future audio foundation models [Défossez et al., 2024, Gong et al., 2026]. MiniMind-O keeps Mimi’s eight-codebook representation. The difference is where the predictor lives: the audio-code predictor is attached to a very small omni model rather than delegated to a large standalone acoustic model. 

Multimodal feature alignment. For vision-language modeling, CLIP and BLIP-2 established a practical separation between perception and language modeling: a frozen or slowly changing encoder produces features, and a bridge maps them into the LLM space [Radford et al., 2021, Li et al., 2023]. LLaVA, Qwen-VL, Qwen2-VL, and SigLIP2 refine this encoder-side foundation with stronger visual representations and instruction-tuned multimodal use cases [Liu et al., 2024, Bai et al., 2023, Wang et al., 2024, Tschannen et al., 2025]. The MiniMind line has used the same minimal-recipe philosophy in its language-only and vision-language variants [Gong, 2024, 2025]. In the current MiniMind-O codebase, both audio and vision use plain two-layer MLP projectors. This is a simpler choice: the external encoders carry perception, and the projectors only have to map their hidden states into the MiniMind embedding space. 

## 3 Model Architecture

Figure 1 shows the data path implemented in model_omni.py. Text enters through the native token embedding table. Speech is converted to SenseVoice frontend features and passed through a frozen SenseVoice encoder; the resulting states are mapped by MMAudioProjector, a two-layer MLP with LayerNorm and GELU. Images are encoded by a frozen SigLIP2 vision model and mapped by the same kind of MLP projector. The projected states preserve the encoder sequence axis and replace contiguous <|audio_pad|> or <|image_pad|> embedding positions in the Thinker input sequence. 

The Thinker is the full MiniMind transformer, while the Talker is an additional module with num_talker_hidden_layers=4 MiniMind blocks, its own RMSNorm, Mimi-code embedding, codec projection, and audio-code heads. When loading a MiniMind checkpoint that has no Talker weights and the hidden sizes match, the Talker blocks are initialized by copying the last four Thinker blocks. During forward propagation, the Talker input is the sum of two projected streams: embed_proj(bridge_states) scaled by a learned text scale, and codec_proj(talker_emb) scaled by a learned audio scale. The module therefore reads both semantic states and autoregressive Mimi-code history instead of serving as a simple suffix of the language model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/59b6b1a7a8294c8fddeca1ebb34fa8faddc546fbeb5d476c1e641bd6981ffd32.jpg)



Figure 4: Training pipeline used by the current implementation. The active training script runs train_sft_omni.py on T2A, I2T, and A2A data, with all mode for full-model updates and a vision_proj pass for projector-only visual alignment. SenseVoice and SigLIP2 remain frozen during training.


The audio-code input and output interfaces are intentionally non-full-rank. TalkerEmbedding uses one shared embedding table plus per-codebook low-rank adapters, and TalkerHead uses one shared linear head plus per-codebook low-rank adapters. This compact interface is important at 0.1B scale: the model still sees codebook-specific residuals, while the large shared component is not duplicated eight times. 

Figure 2 expands the Talker side. Speaker control is represented in the audio-code buffer rather than the text stream. If a speaker embedding is available, the dataset reserves one position before the reference-code region and fills all eight audio layers at that position with <|audio_spk|>; the model then replaces the Talker embedding at that position with a projected 192-dimensional CAM++ vector. Reference Mimi codes are right-aligned before the target speech region and are masked from the audio loss. This layout makes the reference act as a prompt rather than a reconstruction target, which matters when the same voice has to be reused for a different sentence. 

Appendix Table 6 lists each module, its concrete model, key configuration, and parameter count. The trainable counts deduplicate the tied MiniMind token embedding and text lm_head. The evaluation tables keep the experiment-level checkpoint accounting, so they should be read as the model-size labels used for comparison rather than as a decomposition of that table. 

## 3.1 Middle-layer Bridge

A small omni model is sensitive to the bridge layer. The embedding layer still mainly contains token identity and injected multimodal features; it has not accumulated enough context for pronunciation, syntax, or cross-modal reference. The last layer has the opposite bias. It is already shaped by the next-text-token classifier, so the hidden state carries the geometry and token-selection noise of the LM head rather than the acoustic conditions needed by the Talker. In consistency experiments, moving the bridge too deep increases Talker CER, which is a sign that the acoustic path is being conditioned on states already over-specialized for text logits. 

MiniMind-O therefore extracts the bridge state from a middle Thinker layer, by default num_hidden_layers // 2 - 1. The choice is close in spirit to the middle-layer hidden extraction used in Qwen-Omni-style Thinker–Talker systems [Xu et al., 2025a,b]. In the default eight-layer MiniMind setting, this means the bridge is captured after layer 3. A learned embed_proj maps this state into the Talker hidden space before it is fused with codec-history features. The 768-dimensional Talker is kept because the ablation in Table 2 shows that narrower variants lose consistency before the parameter saving becomes useful. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/6682934e1535cbe750760552ceebb58202e4bfc581eafdc0b5b0984525f3cc7f.jpg)



Figure 5: Input token layout in MiniMind-O. Text tokens, audio placeholders, image placeholders, speaker tokens, reference codes, and target audio codes occupy aligned positions so that the Thinker and Talker can be trained under a single autoregressive schedule.


## 4 Sequence Format and Streaming Decoding

Figure 3 and Figure 5 show the actual sequence layout. Each training example is a nine-stream sequence: eight audio-code streams plus one text stream. The Thinker reads the text stream, where repeated audio or image placeholders mark positions to be replaced by projected SenseVoice or SigLIP2 states. The Talker reads the eight audio streams. Before the assistant response, the audio streams are padded, optionally filled with right-aligned reference codes, and optionally marked with a speaker-token position. After the response starts, they carry target Mimi codes. Only the target region receives audio labels; reference and conditioning positions stay masked. 

For a response with text tokens $y _ { 1 : T }$ and Mimi code matrix $\mathbf { a } \in \mathbb { N } ^ { 8 \times T ^ { \prime } }$ , MiniMind-O optimizes a joint next-token objective, 

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{text}} + \lambda_ {\mathrm{audio}} \sum_ {q = 1} ^ {8} \mathcal {L} _ {\mathrm{audio}} ^ {(q)},\tag{1}
$$

where $q$ indexes the Mimi codebook layer. Invalid or conditioning-only positions are masked. The dataset staggers audio targets by codebook layer: layer q starts at assistant_start $^ + \texttt { q + 1 }$ . In streaming inference, the first generated text step has no audio output, and the eight codec layers become available with the same delayed schedule. Once a complete eight-layer frame is available, the Mimi codes can be decoded incrementally into 24 kHz waveform, so playback can begin before the full textual response is complete. 

This format makes the evaluation stricter than a cascaded ASR–LLM–TTS system in one specific sense. The Talker is judged against the Thinker’s own text, not against an external transcript or a hand-written reference. When numerals, rare names, or longer clauses are not rendered correctly, the mismatch can be traced back to the shared omni path. A large standalone TTS module may absorb part of this difficulty; here the behavior remains visible. 

## 5 Training Pipeline

The current training entry is train_sft_omni.py. Its mode switch is deliberately small: all updates the trainable MiniMind/Talker/projector parameters together, audio_proj freezes the rest of the model and trains only the audio projector, and vision_proj does the same for the vision projector. The active train.sh runs full-model passes over sft_t2a, sft_i2t, and sft_a2a, followed by a projector-only sft_i2t pass, for both dense and MoE variants. This differs from the older README description that names separate t2t, t2a, and a2a modes; those names describe the data type, not the current command-line mode interface. 

All runs reported in this paper are produced on a single workstation with four NVIDIA RTX 3090 GPUs (24 GB each), using PyTorch DDP launched via torchrun –nproc_per_node 4. Training uses bf16 mixed precision with the AdamW optimizer, a per-GPU batch size of 32, no gradient accumulation, and gradient clipping at 1.0. The full-model T2A pass uses learning rate $5 \times \mathrm { { 1 0 ^ { - 6 } } }$ for one epoch on sft_t2a; the audio-projector A2A pass and the vision-projector I2T pass use $5 \times 1 0 ^ { - 4 }$ and ${ \bar { 5 } } \times 1 0 ^ { - 5 }$ respectively for one epoch each; the full-model $\mathbf { A } 2 \mathbf { A }$ pass uses $5 ^ { ^ { \cdot } } \times 1 0 ^ { - 5 }$ for three epochs on sft_a2a; and the full-model I2T pass uses $5 \times 1 0 ^ { - 6 }$ for one epoch with a 768-token context. Wall-clock time per stage is approximately 45 min for T2A, 25 min for the audio-projector A2A pass, 75 min for the three-epoch A2A pass, and 45 min for each I2T pass, so a complete dense or MoE training cycle finishes in under four hours on this setup. Working at 0.1B active scale is what makes this consumer-GPU schedule feasible: at frontier scale, the same loop would not be reproducible without a much larger compute budget. 


Table 1: Main training datasets used by MiniMind-O. Audio durations are computed from the preextracted Mimi-code statistics in the released dataset.


<table><tr><td>Dataset</td><td>Items</td><td>Input speech</td><td>Output speech</td><td>Total speech</td></tr><tr><td>sft_i2t</td><td>~100K</td><td>-</td><td>-</td><td>-</td></tr><tr><td>sft_t2a</td><td>1,248,923</td><td>-</td><td>1636.01 h</td><td>1636.01 h</td></tr><tr><td>sft_a2a</td><td>414,024</td><td>1711.97 h</td><td>423.40 h</td><td>2135.37 h</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/7c8f9667d28f4a6dd4996a9133bb66ec405362b84c2a7231c2743a2f6702ecbc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/f41e7ee4e0b96a63804296c32295ccf7e5c9db8ba36b3b7d0754f7a7f135ecf9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/701a027c63983c80ae007464f583c91b202d61fe5c157a867ba1cb1b602e9b62.jpg)



Figure 6: Text-to-audio training curves for minimind-3o and minimind-3o-moe. The plotted curve uses the cleaned log segment after removing the erroneous resume interval caused by loading an incompatible checkpoint.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/6ce43ed644dfe380e9131a964f71bdd02b5edffc95550faed6918495838bb5b0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/5d6f37753b488ea5de874563c70fa4ebaa5de132ff67094fa52d492333b18cc4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/8eb92774b990f9a5ee5f366fcad5c5fb35562224e0cf67cdfdad10637270348a.jpg)



Figure 7: Audio-to-audio training curves for minimind-3o and minimind-3o-moe. The A2A stage is trained after text-to-audio learning and exposes the full speech-in/speech-out loop.


Table 1 gives the data scale used in the release. The public dataset is part of the contribution because it fixes the exact sequence and codec layout used by the model rather than leaving reproduction to a private preprocessing pipeline. sft_t2a contains 1,248,923 samples and 1636.01 h of output speech. sft_a2a contains 414,024 samples, 1711.97 h of input speech, and 423.40 h of output speech. The text-to-audio split is close to balanced between Chinese and English outputs, with 45.7% Chinese, 46.5% English, and 7.8% mixed content. The audio-to-audio split is Chinese-heavy: 70.8% Chinese, 21.2% English, and 8.0% mixed content. This distribution shows up in behavior. Short Chinese and English replies are usually stable; longer English speech is where pronunciation drift and omissions become easier to trigger. 

Figure 6 and Figure 7 show the two speech-generation stages. The T2A curve uses the cleaned log segment; an earlier resume from an incompatible checkpoint produced a loss spike, and that interval is not used here. The MoE variant has a larger total parameter count but roughly the same active scale as the dense model, so these curves are more useful for reading capacity allocation than for claiming equal-compute superiority. 

Figure 8 isolates the Talker-side low-rank interfaces from the rest of the model. The experiment freezes the Thinker and varies the rank of the TalkerEmbedding and TalkerHead adapters on the same A2A subset. Increasing the unified rank improves convergence, final audio loss, and codebook accuracy, but the gain becomes gradual once the adapter reaches a few million parameters. The decoupled runs are more diagnostic: increasing the TalkerHead rank from 16 to 256 gives a larger improvement than increasing the TalkerEmbedding rank under the same setting. This matches the roles of the two interfaces. The embedding side mainly reads recent Mimi-code history, while the head side has to separate eight codebook distributions over the full audio vocabulary. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/5391071f51088bca93403d89d3edab3322cf901c51a06b5f0fc7e8a5134e24b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/1f7ccbc2301d5f31d7df920c0aa88fd04527d64980199d8ee8bba70baa4387bf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/a1065fd1eba5b2bc359e0c8ccb1e209d29fd4612a61c0bf5e2546a848c5dd8cb.jpg)



(d)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/815f43449a629c67a0e6533fc5f5e1e46da8d0abb8765f8653eaded3e8a20c15.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/c85a79c79bb8698cc59756dca6cb683c08ca9e83e47c2309eaf1572ce1b0c11e.jpg)



(f)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/d017a44c640e05f38036666e85cddc4eea8ee97c2cbdbf8432c26bc4b6147d8b.jpg)



Figure 8: Rank ablation for the Talker-side low-rank interfaces. The top row sweeps a unified rank for TalkerEmbedding and TalkerHead; the bottom row decouples the two ranks. Solid curves or bars report audio loss, while dashed curves or overlaid markers report audio accuracy. The results show that moderate ranks already recover most of the parameter-efficient gain, and that the output head rank is more important than the embedding rank.



Table 2: Talker hidden-size ablation. The 768-dimensional Talker is selected for both variants because it gives the best average CER and keeps the Thinker–Talker dimensional interface simple.


<table><tr><td>Variant</td><td>Talker hidden</td><td>Params</td><td>Avg CER ↓</td><td>Short ↓</td><td>Mid / Long ↓</td></tr><tr><td>Dense</td><td>768</td><td>115.29M</td><td>0.0897</td><td>0.1528</td><td>0.0874 / 0.0675</td></tr><tr><td>Dense</td><td>512</td><td>96.13M</td><td>0.1745</td><td>0.2709</td><td>0.2455 / 0.0976</td></tr><tr><td>Dense</td><td>384</td><td>88.72M</td><td>0.2767</td><td>0.3904</td><td>0.1865 / 0.4046</td></tr><tr><td>MoE</td><td>768</td><td>317.05M-A115.33M</td><td>0.0900</td><td>0.2075</td><td>0.0533 / 0.0271</td></tr><tr><td>MoE</td><td>512</td><td>261.32M-A96.17M</td><td>0.1265</td><td>0.0711</td><td>0.1490 / 0.1464</td></tr><tr><td>MoE</td><td>384</td><td>240.04M-A88.75M</td><td>0.3280</td><td>0.3757</td><td>0.2777 / 0.4313</td></tr></table>

## 6 Evaluation

The evaluation is built around consistency properties that are easy to miss in demos. For each prompt, the model produces Thinker text and Talker audio. The audio is transcribed by Qwen3-ASR-Flash, and the transcript is compared with the Thinker text. The internal consistency runs report CER, while the cross-model English and vision-language comparisons additionally report WER. These metrics leave naturalness and preference to separate evaluation; here they ask a narrower question: after the Talker turns the hidden state into waveform, does the spoken or written output still match the intended text? The protocol is therefore ASR-dependent and should not be read as a MOS or preference study. In particular, numeral formatting can inflate edit distance when the waveform is correct but the ASR writes a number in words. 

Table 2 reports the Talker hidden-size ablation. The 768-dimensional setting is the only one that stays stable for both dense and MoE variants. Reducing the Talker to 512 or 384 does save parameters, but it also narrows the acoustic state seen by each codebook head. Since Mimi prediction is an eight-layer problem, the bottleneck is amplified across codebooks. The ablation rules out a simple scaling assumption: the Talker cannot be made very thin just because the semantic plan comes from the Thinker. 


Table 3: Voice-cloning similarity measured by CAM++ speaker embeddings [Wang et al., 2023b]. The baseline row refers to the earlier reference-code-only setting reported during development.


<table><tr><td>Model</td><td>Seen ↑</td><td>Unseen ↑</td><td>Overall ↑</td></tr><tr><td>Previous baseline</td><td>0.6150</td><td>0.5310</td><td>-</td></tr><tr><td>minimind-3o</td><td>0.6472</td><td>0.5654</td><td>0.5995</td></tr><tr><td>minimind-3o-moe</td><td>0.6267</td><td>0.5702</td><td>0.5937</td></tr></table>


Table 4: Cross-model English T2A consistency under the same brief-answer constraint. minimind-3o is smaller than Mini-Omni and Mini-Omni2 [Xie and Wu, 2024a,b], but the gap is concentrated in medium-length answers.


<table><tr><td>Model</td><td>Params</td><td>Avg CER ↓</td><td>Avg WER ↓</td></tr><tr><td>Mini-Omni</td><td>0.5B</td><td>0.0101</td><td>0.0185</td></tr><tr><td>Mini-Omni2</td><td>0.5B</td><td>0.0371</td><td>0.0431</td></tr><tr><td>minimind-3o</td><td>0.1B</td><td>0.0964</td><td>0.0973</td></tr></table>


Table 5: Vision-language comparison with length-matched references generated by Qwen-VL-Plus [Bai et al., 2023]. CER/WER are high because open-ended image descriptions admit many valid paraphrases.


<table><tr><td>Model</td><td>Params</td><td>Avg CER ↓</td><td>Avg WER ↓</td></tr><tr><td>Mini-Omni2</td><td>0.5B</td><td>0.7609</td><td>0.9756</td></tr><tr><td>minimind-3o</td><td>0.1B</td><td>0.8241</td><td>1.0293</td></tr></table>

Table 3 shows the voice-cloning evaluation; the per-speaker breakdown is in Appendix Table 7. The seen split uses the five built-in voices shipped in voices.pt: dylan, eric, serena, uncle_fu, and vivian. The unseen split uses seven prompts from voices_unseen.pt: arthur, chelsie, cherry, ethan, jennifer, momo, and moon. For each voice, generation keeps the same textual questions and changes only the in-context speaker condition, namely the reference Mimi codes and the 192-dimensional CAM++ vector. Dense is slightly better on seen speakers, and MoE is slightly better on unseen speakers, but the overall gap is small. Both improve over the earlier reference-code only baseline, from 0.6150 to 0.6472 on seen voices for the dense model and from 0.5310 to 0.5702 on unseen voices for the MoE model. The per-speaker table shows that the best individual voices (uncle_fu, serena, arthur) exceed 0.70 cosine similarity for at least one variant, while the low outliers (eric under MoE, moon under dense) usually coincide with degraded generated audio before the speaker encoder is applied. 

Table 5 reports a small vision-language comparison. Mini-Omni does not support this path, so the comparison is between Mini-Omni2 and minimind-3o [Xie and Wu, 2024a,b]. The evaluation uses nine synthetic images; for each output, Qwen-VL-Plus generates a separate length-matched reference [Bai et al., 2023]. The absolute values are high because open-ended image descriptions admit many valid paraphrases. Under the same protocol, minimind-3o trails Mini-Omni2 but remains in the same order of magnitude while using about one fifth of the parameters. Per-sample values are in Appendix Table 10. 

## 7 Discussion and Limitations

The main lesson from MiniMind-O is that the omni loop has a meaningful small-model regime. A full text–speech–image loop can be made public and inspectable at roughly 0.1B active parameters; the training data can be released in a form that preserves the actual multimodal layout; the eight-codebook embedding/head interface does not have to be fully duplicated across codebooks; and a middle-layer bridge gives the Talker a cleaner semantic condition than the final next-token-prediction state. These are positive results even though the model remains far from frontier-scale systems. 

The limitations are also clear. Speech naturalness and long-form stability remain behind larger speech-text models, with medium-length English answers being the most visible weak point. The visual pathway uses a frozen SigLIP2 encoder, 64 placeholder positions, and a plain MLP projector, so its role is closer to a compact vision-to-speech path than to a large-VLM replacement. Voice cloning improves over the earlier reference-code-only baseline, while still depending heavily on reference quality and on whether the generated audio is clean enough for the speaker encoder to read. The MoE variant is best read as a capacity-allocation experiment rather than a final expert layout. The evaluation is also deliberately narrow: the main automatic scores measure transcript consistency, not human naturalness, latency under load, safety behavior, or robustness to noisy far-field speech. 

The claim is intentionally narrow. MiniMind-O is not presented as a competitor to frontier-scale systems; its value is that the complete omni loop can be reproduced and inspected without hiding the key choices behind scale. 

## 8 Conclusion

This report introduced MiniMind-O, a 0.1B-scale open omni model with text, speech, and image inputs and streaming speech output. The current code combines a full MiniMind Thinker, an independent four-layer Talker, middle-layer semantic bridging, MLP-based audio/vision projection, Mimi-code speech generation, and staged SFT over released T2A, I2T, and A2A data. The dense and MoE variants both maintain usable Thinker–Talker consistency under short-answer settings, support speaker-conditioned generation, and run basic vision-language-to-speech interaction. The broader message is that small omni models can serve as controlled research artifacts: with public data, a middle hidden bridge, and low-rank codebook-specific embedding/head adapters, a complete loop can be made parameter-efficient enough to study directly. In this sense, MiniMind-O contributes a reproducible small-scale baseline for analyzing speech-native omni design, not only a runnable demo. The remaining gaps are exposed by the same implementation, which makes the small regime useful for analysis rather than only for deployment efficiency. 

## References



Keyu An et al. Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms. arXiv preprint arXiv:2407.04051, 2024. 





Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 





Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Défossez. Simple and controllable music generation. Advances in Neural Information Processing Systems, 36, 2024. 





Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022. 





Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024. 





Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. Llama-omni: Seamless speech interaction with large language models. arXiv preprint arXiv:2409.06666, 2024. 





Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 





Jingyao Gong. Minimind: Train a small language model from scratch. https://github.com/ jingyaogong/minimind, 2024. 





Jingyao Gong. Minimind-v: Train a small vision-language model from scratch. https://github. com/jingyaogong/minimind-v, 2025. 





Jingyao Gong. Minimind-o: Train a tiny omni model from scratch. https://github.com/ jingyaogong/minimind-o, 2026. 





Yitian Gong, Kuangwei Chen, Zhaoye Fei, Xiaogui Yang, Ke Chen, Yang Wang, Kexin Huang, Mingshu Chen, Ruixiao Li, Qingyuan Cheng, et al. Moss-audio-tokenizer: Scaling audio tokenizers for future audio foundation models. arXiv preprint arXiv:2602.10934, 2026. 





Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, et al. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946, 2025. 





Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023. 





Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, et al. Baichuan-audio: A unified framework for end-to-end speech interaction. arXiv preprint arXiv:2502.17239, 2025. 





Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 





Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussà, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, et al. Spirit-lm: Interleaved spoken and written language model. Transactions of the Association for Computational Linguistics, 13:30–52, 2025. 





Openai. https://openai.com/index/hello-gpt-4o/, 2024. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 





Hubert Siuzdak. https://github.com/hubertsiuzdak/snac/, 2024. 





Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 





Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023a. 





Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, and Qian Chen. Cam++: A fast and efficient network for speaker verification using context-aware masking. arXiv preprint arXiv:2303.00332, 2023b. 





Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 





Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024a. 





Zhifei Xie and Changqiao Wu. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. arXiv preprint arXiv:2410.11190, 2024b. 





Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025a. 





Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report. https://arxiv.org/abs/2509.17765, 2025b. 





Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612, 2024. 



## Appendices

## A. Module and Evaluation Details

This appendix collects the detailed tables referenced in the main text. Table 6 enumerates every module in the current MiniMind-O implementation together with its concrete model, key hyperparameters, and parameter count. The trainable counts deduplicate the tied MiniMind token embedding and text lm_head; frozen modules are loaded as-is and never updated during training. 


Table 6: Main modules used by the current implementation. Trainable component counts are taken from the current PyTorch modules; external perception and codec models are frozen and are not counted as active MiniMind-O parameters.


<table><tr><td>Module</td><td>Concrete model or layer</td><td>Key configuration</td><td>Status / params (Dense / MoE)</td></tr><tr><td>Thinker</td><td>MiniMind Transformer</td><td>8 layers, hidden 768, 8 query heads, 4 KV heads, vocab 6400</td><td>trainable, 63.91M / 198.42M</td></tr><tr><td>Talker</td><td>independent Mini-Mind blocks</td><td>4 layers, hidden 768, audio vocab 2112, 8 codebook heads, rank-256 embedding/head adapters</td><td>trainable, 47.05M / 114.30M</td></tr><tr><td>Audio projector</td><td>MMAudioProjector</td><td>LayerNorm(512) – Linear – GELU – Linear to hidden 768</td><td>trainable, 0.99M</td></tr><tr><td>Vision projector</td><td>MMVisionProjector</td><td>LayerNorm(768) – Linear – GELU – Linear to hidden 768</td><td>trainable, 1.18M</td></tr><tr><td>Audio encoder</td><td>SenseVoice-Small</td><td>50 encoder blocks, output size 512, 16 kHz frontend</td><td>frozen, 234.00M</td></tr><tr><td>Vision encoder</td><td>SigLIP2 base patch32-256</td><td>12 layers, hidden 768, 12 heads, 64 image tokens</td><td>frozen, 94.55M</td></tr><tr><td>Speech codec</td><td>Mimi</td><td>8 codebooks, size 2048, 12.5 Hz frames, 24 kHz waveform</td><td>frozen, 96.15M</td></tr><tr><td>Speaker condition</td><td>CAM++ embedding</td><td>192-dimensional vector projected by spk_proj</td><td>precomputed, no online encoder</td></tr></table>

Table 7 breaks down voice-cloning similarity by individual speaker. The five seen voices are the built-in prompts shipped in voices.pt; the seven unseen voices come from voices_unseen.pt and are never seen during training. For each voice the same set of textual questions is used, changing only the in-context speaker condition (reference Mimi codes and 192-dimensional CAM++ vector). The best individual voices (uncle_fu, serena, arthur) exceed 0.70 cosine similarity for at least one variant, while the lowest outliers (eric under minimind-3o-moe, moon under minimind-3o) typically coincide with degraded generated audio quality before the speaker encoder is applied. 

Tables 8 and 9 expand the cross-model English T2A comparison from the main text. All three models receive the same instruction (Answer briefly in one short sentence). The length-bucket view (Table 8) shows that minimind-3o is competitive with Mini-Omni2 on short answers (≤15 words) but falls behind on medium-length responses (16–30 words), where the Talker must sustain pronunciation and lexical consistency across a full clause. 

The per-question breakdown (Table 9) reveals that 14 out of 20 questions achieve zero CER for all three models. The few high-CER outliers are mainly driven by surface-form mismatches rather than clear pronunciation failures. For example, question 04 involves the number “299,792,458”, while the ASR may transcribe the spoken answer as “two hundred ninety-nine million. . . ”, inflating character-level distance. Question 13 shows the same metric sensitivity for named entities, where a small transcript variation can dominate the score for a short answer. 

Table 10 gives the per-sample vision-language results. Mini-Omni does not support this path, so the comparison is limited to Mini-Omni2 and minimind-3o. Each image is described independently; Qwen-VL-Plus generates a separate length-matched reference for the same image, and CER/WER are computed against that reference. The absolute values are high across both models because open-ended image descriptions admit many valid paraphrases and detail orderings—two correct descriptions of the same image can share very few exact n-grams. Under this protocol minimind-3o trails Mini-Omni2 but stays within the same range while using about one fifth of the parameters. 


Table 7: Per-speaker voice-cloning similarity measured by CAM++ cosine similarity. The seen speakers are the built-in prompts shipped with the release; unseen speakers are held out from the default voice set.


<table><tr><td>Split</td><td>Speaker</td><td>minimind-3o ↑</td><td>minimind-3o-moe ↑</td></tr><tr><td>Seen</td><td>dylan</td><td>0.6997</td><td>0.6837</td></tr><tr><td>Seen</td><td>eric</td><td>0.5289</td><td>0.4232</td></tr><tr><td>Seen</td><td>serena</td><td>0.7092</td><td>0.7041</td></tr><tr><td>Seen</td><td>uncle_fu</td><td>0.7241</td><td>0.7337</td></tr><tr><td>Seen</td><td>vivian</td><td>0.5744</td><td>0.5888</td></tr><tr><td>Unseen</td><td>arthur</td><td>0.7171</td><td>0.6750</td></tr><tr><td>Unseen</td><td>chelsie</td><td>0.6437</td><td>0.6240</td></tr><tr><td>Unseen</td><td>cherry</td><td>0.5689</td><td>0.5678</td></tr><tr><td>Unseen</td><td>ethan</td><td>0.4783</td><td>0.4847</td></tr><tr><td>Unseen</td><td>jennifer</td><td>0.4749</td><td>0.4003</td></tr><tr><td>Unseen</td><td>momo</td><td>0.6470</td><td>0.5720</td></tr><tr><td>Unseen</td><td>moon</td><td>0.4282</td><td>0.6673</td></tr></table>


Table 8: Length-bucket breakdown for the cross-model English T2A comparison. Each entry reports CER / WER with the number of evaluated samples in parentheses.


<table><tr><td>Length bucket</td><td>Mini-Omni</td><td>Mini-Omni2</td><td>minimind-3o</td></tr><tr><td>Short (≤ 15 words)</td><td>0.0195 / 0.0384 (n=8)</td><td>0.0503 / 0.0584 (n=14)</td><td>0.0531 / 0.0417 (n=8)</td></tr><tr><td>Mid (16–30 words)</td><td>0.0038 / 0.0052 (n=12)</td><td>0.0062 / 0.0076 (n=6)</td><td>0.1327 / 0.1420 (n=11)</td></tr><tr><td>Long (31–60 words)</td><td>-</td><td>-</td><td>0.0431 / 0.0508 (n=1)</td></tr></table>


Table 9: Per-question cross-model English T2A comparison. Each cell reports CER / WER. Questions are abbreviated; all are prefixed with “Answer briefly in one short sentence.” Entries with CER > 0.3 are typically caused by surface-form ASR mismatches such as number spelling or named-entity variants.


<table><tr><td>#</td><td>Question (abbreviated)</td><td>Mini-Omni</td><td>Mini-Omni2</td><td>minimind-3o</td></tr><tr><td>00</td><td>Hello, how are you today?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>01</td><td>Can you tell me a joke?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>02</td><td>What is the capital of France?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>03</td><td>How do you make a cup of coffee?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>04</td><td>What is the speed of light?</td><td>0.000 / 0.000</td><td>0.382 / 0.286</td><td>1.410 / 1.471</td></tr><tr><td>05</td><td>Explain what AI is.</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>06</td><td>Tallest mountain in the world?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>07</td><td>How many planets in the solar system?</td><td>0.156 / 0.125</td><td>0.303 / 0.250</td><td>0.000 / 0.000</td></tr><tr><td>08</td><td>What causes rainbows?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>09</td><td>Recommend a good book?</td><td>0.000 / 0.182</td><td>0.000 / 0.182</td><td>0.000 / 0.000</td></tr><tr><td>10</td><td>Largest ocean on Earth?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>11</td><td>How does photosynthesis work?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.017 / 0.045</td></tr><tr><td>12</td><td>Benefits of regular exercise?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>13</td><td>Who invented the telephone?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.425 / 0.333</td></tr><tr><td>14</td><td>Meaning of life?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>15</td><td>How do airplanes stay in the air?</td><td>0.000 / 0.000</td><td>0.019 / 0.100</td><td>0.033 / 0.045</td></tr><tr><td>16</td><td>Virus vs. bacteria?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>17</td><td>Explain blockchain technology.</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td>18</td><td>Three tips for better sleep?</td><td>0.045 / 0.063</td><td>0.037 / 0.045</td><td>0.043 / 0.051</td></tr><tr><td>19</td><td>Why is the sky blue?</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td><td>0.000 / 0.000</td></tr><tr><td colspan="2">Average</td><td>0.010 / 0.019</td><td>0.037 / 0.043</td><td>0.096 / 0.097</td></tr></table>


Table 10: Per-sample vision-language comparison. Each cell reports output length / reference length / CER / WER.


<table><tr><td>#</td><td>Mini-Omni2</td><td>minimind-3o</td></tr><tr><td>00</td><td>158 / 128 / 0.7976 / 1.0391</td><td>111 / 103 / 0.7883 / 0.9709</td></tr><tr><td>01</td><td>260 / 233 / 0.7273 / 0.9914</td><td>109 / 87 / 0.8013 / 1.0920</td></tr><tr><td>02</td><td>110 / 89 / 0.7957 / 0.9888</td><td>126 / 98 / 0.8728 / 1.0816</td></tr><tr><td>03</td><td>79 / 84 / 0.7408 / 0.9286</td><td>113 / 104 / 0.8031 / 0.9519</td></tr><tr><td>04</td><td>94 / 102 / 0.7273 / 0.8529</td><td>130 / 101 / 0.8531 / 1.0594</td></tr><tr><td>05</td><td>91 / 72 / 0.7209 / 1.0556</td><td>115 / 91 / 0.8629 / 1.0989</td></tr><tr><td>06</td><td>187 / 194 / 0.7293 / 0.9021</td><td>119 / 118 / 0.7551 / 0.8814</td></tr><tr><td>07</td><td>295 / 267 / 0.7682 / 0.9625</td><td>114 / 89 / 0.8548 / 1.0449</td></tr><tr><td>08</td><td>143 / 118 / 0.8414 / 1.0593</td><td>137 / 109 / 0.8254 / 1.0826</td></tr></table>

## B. Qualitative Examples

This appendix shows representative outputs from the three interaction modes supported by MiniMind O: real-time streaming with barge-in interruption (Figure 9), audio-to-audio dialogue (Figure 10), and image-conditioned speech generation (Figure 11). The examples are generated by the minimind-3o variant, and the HTML demo page bundled with the release includes playable audio for the displayed cases. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/651f349acfd29de5a33451027be8e2c98ab7258bc032393078fd38fbda7cee90.jpg)



Figure 9: Real-time interaction interface. Streaming speech generation allows playback while decoding continues, and VAD-triggered barge-in can stop the current output when a new user turn is detected.


Figure 9 shows the real-time streaming and barge-in interaction setting. After the user finishes speaking, the Thinker first performs the semantic-side prefill, the Talker starts producing audio codes, and the Mimi decoder writes the 24 kHz waveform as new code frames become available. The lower timeline illustrates the barge-in path: when the user speaks again during model playback, the system detects the new speech event, abandons the current generation, and begins a fresh prefill–reply cycle. This is not a claim of human-level full-duplex turn taking; the interrupt detection is still based on a simple VAD threshold rather than semantic understanding of overlap. It is a smaller but practically useful engineering loop: the system can leave the speaking state, accept a new request, and produce the next response without waiting for the previous waveform to finish. 

Figure 10 shows audio-to-audio cases where real speech is used as input and the model returns both text and speech. Short assistant-style dialogue is the most stable setting: the Thinker produces a compact semantic answer, and the Talker can render it before audio-code errors accumulate. Chinese explanatory prompts usually remain coherent, while English responses show more variation in pronunciation and rhythm. Longer answers are still possible, but they expose the same weakness as 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/7a0914340a9fc86c6019af3738d61645092e5f4af28ca699ead4b1b248090271.jpg)



Figure 10: Qualitative A2A examples. The model receives speech input and returns aligned text and speech output, exposing the full speech-in/speech-out loop.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-30/844f80f5-4b65-4168-804e-b53c2fa40779/6712e87dfa9e819ac6e187b9405926f637c729f5fbebe5b9290bc9e003927ffc.jpg)



Figure 11: Image-to-audio qualitative examples. Image features are projected into the Thinker, and the resulting answer can be rendered through the Talker as speech.


Table 4: pronunciation drift and small word omissions become easier to trigger as the acoustic path has to sustain a longer sentence. 

Figure 11 illustrates image-conditioned speech generation. The path connects visual encoding, text generation, and speech rendering in a single pipeline: SigLIP2 provides image features, the projector maps them into the Thinker space, and the Talker renders the resulting answer as speech. The examples show that the pipeline can condition speech on image content, but they also expose typical small-model errors: some outputs capture the coarse scene, while others replace the main object or confuse attributes, such as animal categories or vehicle type. These errors are consistent with the 64 image-placeholder budget and the 0.1B base scale, so the examples should be read as evidence that the small omni pipeline runs end-to-end rather than as an upper bound on open-ended image description. 