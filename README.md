## SAEs on Individual Attention Head Outputs

SAEs seem to work well on individual attention head outputs. I trained SAEs on concatenated outputs with **block-diagonal** encoder and decoder weight matrices. The code to train these models is in [this branch of my fork of SAELens](https://github.com/shehper/SAELens/tree/block_diag_sae). 

I trained SAEs on attention layer outputs of gelu-2l layer-1 and gpt2-small layers 4 and 5. In all cases, SAEs learned interpreable features. 

Training block-diagonal SAEs have many advantages. 
- With the same expansion factor, the SAEs contain $\sim 1/{n_{\text{heads}}}$ times the number of trainable parameters as a fully dense SAE as trained by Kissane et al. This leads to cheaper costs of training and inference.
- Each feature is by-design attributed to a single head. We do not compute "direct feature attribution" to get the attribution of a feature to each head, and hence do not worry "whether a multi-head feature is exhibiting attention head superposition".
- When given a task / prompt, we are able to tell the role that each head plays by looking at the single-head features that activate in that task / prompt. 
- We can check with more certainty the claims of whether a head is "moonosemantic". If we find a feature that does not have the same interpretation as all of the other features in a head, the head is polysemantic. 

I include W&B logs and feature dashboards of the two gpt2-small SAEs.
- gpt2-small layer 4 [W&B](https://wandb.ai/shehper/gpt2-small-attn-4-sae/runs/pumu7rz3?nw=nwusershehper) [Dashboards](https://shehper.github.io/attn_saes/layer_4.html)
- gpt2-small layer 5 [W&B](https://wandb.ai/shehper/gpt2-small-attn-5-sae/runs/s4om7ilc?nw=nwusershehper) [Dashboards](https://shehper.github.io/attn_saes/layer_5.html)

A dashboard file contains feature dashboards of the first 10 features from each head. To look through them all, click on the top-left corner to change the number. Features 0-9 belong to head 0, 2048-2057 belong to head 1, and so on. 


## Reproduction

`pip install` the branch of SAELens mentioned above. Then do

```
python -u train_gpt_block_diag.py --l1_coefficients=[2] --hook_layer=4 --expansion_factor=32 --total_training_steps=50000
python -u train_gpt_block_diag.py --l1_coefficients=[3] --hook_layer=5 --expansion_factor=32 --total_training_steps=100000
```

Next, use `gpt_block_diag_dashboards.ipynb` in this repo to load and analyze dashboards. 

## TODO

- The two SAEs have higher L0-norm values (460 and 278) than what is usually reported. It is possible that they are undertrained as the overall loss and the L0-norm were still decreasing when the training stopped. It is possible that with the block-diagonal architecture, L0-norm per head (~38 and 22 respectively) and not the full L0-norm that needs to be small.

- Include "direct feature attribution by source position" in the feature dashboards, following Kissane et al. This will be needed to fully interpret features in attention head outputs. 