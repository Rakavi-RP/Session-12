# Text Generation with GPT

A custom GPT model trained on text data and deployed on Hugging Face Spaces for text generation.

## Repository Structure

```
├── A12.ipynb           # Training notebook with model implementation
├── app.py              # Gradio interface for model deployment
├── assets/            # Images and media files
│   └── Screenshot.png # Interface screenshot
├── best_model.pt       # Trained model weights (523 MB)
├── input.txt           # Training data
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

Note: Due to size limitations, best_model.pt (523 MB) might not be included in the GitHub repository. The model file is deployed directly to Hugging Face Spaces.

## Model Details

- **Architecture**: Custom GPT implementation
- **Parameters**: 124,439,808 parameters
- **BestTraining Loss**: 0.0814


## Model Architecture

The model uses a standard transformer architecture with:
- 12 attention heads
- 12 transformer layers
- 768 embedding dimensions
- GPT-2 tokenizer (50257 tokens vocabulary)

## Training Details

- **Batch Size**: 4
- **Sequence Length**: 128
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW
- **Number of Epochs**: 90
- **Device**: NVIDIA T4 GPU (Google Colab)


## Deployment

The model is deployed on Hugging Face Spaces and can be accessed here:
[GPT Text Generation Space](https://huggingface.co/spaces/Rakavi12/GPT_Text_Generation)

### Interface Screenshot
![GPT Text Generation Interface](path/to/your/Screenshot.png)

### Example Usage

**Input:**
Once upon a time
**Maximum Length:** 30

**Number of Sequences:** 2

**Output:**
Once upon a time he hath
Not belongs.

VIRGILIA:
Then, good madam; I'll doth possess him
My name-master- ring is now mine eyes:
I do thou live, let me alone.


=== Next Sequence ===

Once upon a time he hath
Never for some brown person: I'll make up
Both liking.

VIRGILIA:
Indeed, it is so hot.

MENENIUS:
I'll should hear it must I'll not

## Training Logs 
```
Total parameters: 124,439,808
Epoch 1/90: 100%|██████████| 660/660 [01:45<00:00,  6.26batch/s, loss=5.4918]
Epoch 1 completed. Loss: 5.4918
Best loss - saving model
Epoch 2/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=4.8950]
Epoch 2 completed. Loss: 4.8950
Best loss - saving model
Epoch 3/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=4.5925]
Epoch 3 completed. Loss: 4.5925
Best loss - saving model
Epoch 4/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=4.3560]
Epoch 4 completed. Loss: 4.3560
Best loss - saving model
Epoch 5/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=4.1164]
Epoch 5 completed. Loss: 4.1164
Best loss - saving model
Epoch 6/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=3.9321]
Epoch 6 completed. Loss: 3.9321
Best loss - saving model
Epoch 7/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=3.7683]
Epoch 7 completed. Loss: 3.7683
Best loss - saving model
Epoch 8/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=3.6156]
Epoch 8 completed. Loss: 3.6156
Best loss - saving model
Epoch 9/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=3.4750]
Epoch 9 completed. Loss: 3.4750
Best loss - saving model
Epoch 10/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=3.3363]
Epoch 10 completed. Loss: 3.3363
Best loss - saving model
Epoch 11/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=3.2447]
Epoch 11 completed. Loss: 3.2447
Best loss - saving model
Epoch 12/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=3.1299]
Epoch 12 completed. Loss: 3.1299
Best loss - saving model
Epoch 13/90: 100%|██████████| 660/660 [01:46<00:00,  6.20batch/s, loss=3.0587]
Epoch 13 completed. Loss: 3.0587
Best loss - saving model
Epoch 14/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=2.9454]
Epoch 14 completed. Loss: 2.9454
Best loss - saving model
Epoch 15/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=2.8435]
Epoch 15 completed. Loss: 2.8435
Best loss - saving model
Epoch 16/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=2.7982]
Epoch 16 completed. Loss: 2.7982
Best loss - saving model
Epoch 17/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.7205]
Epoch 17 completed. Loss: 2.7205
Best loss - saving model
Epoch 18/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.6246]
Epoch 18 completed. Loss: 2.6246
Best loss - saving model
Epoch 19/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=2.5396]
Epoch 19 completed. Loss: 2.5396
Best loss - saving model
Epoch 20/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.4499]
Epoch 20 completed. Loss: 2.4499
Best loss - saving model
Epoch 21/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.3935]
Epoch 21 completed. Loss: 2.3935
Best loss - saving model
Epoch 22/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.3740]
Epoch 22 completed. Loss: 2.3740
Best loss - saving model
Epoch 23/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=2.2681]
Epoch 23 completed. Loss: 2.2681
Best loss - saving model
Epoch 24/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.1746]
Epoch 24 completed. Loss: 2.1746
Best loss - saving model
Epoch 25/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.1314]
Epoch 25 completed. Loss: 2.1314
Best loss - saving model
Epoch 26/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=2.0392]
Epoch 26 completed. Loss: 2.0392
Best loss - saving model
Epoch 27/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.9383]
Epoch 27 completed. Loss: 1.9383
Best loss - saving model
Epoch 28/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.8200]
Epoch 28 completed. Loss: 1.8200
Best loss - saving model
Epoch 29/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.6921]
Epoch 29 completed. Loss: 1.6921
Best loss - saving model
Epoch 30/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.6620]
Epoch 30 completed. Loss: 1.6620
Best loss - saving model
Epoch 31/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.4790]
Epoch 31 completed. Loss: 1.4790
Best loss - saving model
Epoch 32/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.3975]
Epoch 32 completed. Loss: 1.3975
Best loss - saving model
Epoch 33/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.3120]
Epoch 33 completed. Loss: 1.3120
Best loss - saving model
Epoch 34/90: 100%|██████████| 660/660 [01:45<00:00,  6.25batch/s, loss=1.1567]
Epoch 34 completed. Loss: 1.1567
Best loss - saving model
Epoch 35/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=1.0422]
Epoch 35 completed. Loss: 1.0422
Best loss - saving model
Epoch 36/90: 100%|██████████| 660/660 [01:45<00:00,  6.26batch/s, loss=0.9954]
Epoch 36 completed. Loss: 0.9954
Best loss - saving model
Epoch 37/90: 100%|██████████| 660/660 [01:45<00:00,  6.25batch/s, loss=0.8542]
Epoch 37 completed. Loss: 0.8542
Best loss - saving model
Epoch 38/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.7791]
Epoch 38 completed. Loss: 0.7791
Best loss - saving model
Epoch 39/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.6613]
Epoch 39 completed. Loss: 0.6613
Best loss - saving model
Epoch 40/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.5888]
Epoch 40 completed. Loss: 0.5888
Best loss - saving model
Epoch 41/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.5474]
Epoch 41 completed. Loss: 0.5474
Best loss - saving model
Epoch 42/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.5108]
Epoch 42 completed. Loss: 0.5108
Best loss - saving model
Epoch 43/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.4622]
Epoch 43 completed. Loss: 0.4622
Best loss - saving model
Epoch 44/90: 100%|██████████| 660/660 [01:45<00:00,  6.25batch/s, loss=0.4213]
Epoch 44 completed. Loss: 0.4213
Best loss - saving model
Epoch 45/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.4053]
Epoch 45 completed. Loss: 0.4053
Best loss - saving model
Epoch 46/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.3448]
Epoch 46 completed. Loss: 0.3448
Best loss - saving model
Epoch 47/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.2981]
Epoch 47 completed. Loss: 0.2981
Best loss - saving model
Epoch 48/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.2970]
Epoch 48 completed. Loss: 0.2970
Best loss - saving model
Epoch 49/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.3033]
Epoch 49 completed. Loss: 0.3033
Epoch 50/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.2819]
Epoch 50 completed. Loss: 0.2819
Best loss - saving model
Epoch 51/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.2678]
Epoch 51 completed. Loss: 0.2678
Best loss - saving model
Epoch 52/90: 100%|██████████| 660/660 [01:45<00:00,  6.25batch/s, loss=0.2669]
Epoch 52 completed. Loss: 0.2669
Best loss - saving model
Epoch 53/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.2346]
Epoch 53 completed. Loss: 0.2346
Best loss - saving model
Epoch 54/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.2218]
Epoch 54 completed. Loss: 0.2218
Best loss - saving model
Epoch 55/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.2377]
Epoch 55 completed. Loss: 0.2377
Epoch 56/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=0.2414]
Epoch 56 completed. Loss: 0.2414
Epoch 57/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=0.1986]
Epoch 57 completed. Loss: 0.1986
Best loss - saving model
Epoch 58/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.1945]
Epoch 58 completed. Loss: 0.1945
Best loss - saving model
Epoch 59/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.2048]
Epoch 59 completed. Loss: 0.2048
Epoch 60/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1914]
Epoch 60 completed. Loss: 0.1914
Best loss - saving model
Epoch 61/90: 100%|██████████| 660/660 [01:46<00:00,  6.23batch/s, loss=0.1780]
Epoch 61 completed. Loss: 0.1780
Best loss - saving model
Epoch 62/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.1693]
Epoch 62 completed. Loss: 0.1693
Best loss - saving model
Epoch 63/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1636]
Epoch 63 completed. Loss: 0.1636
Best loss - saving model
Epoch 64/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1625]
Epoch 64 completed. Loss: 0.1625
Best loss - saving model
Epoch 65/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1528]
Epoch 65 completed. Loss: 0.1528
Best loss - saving model
Epoch 66/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.1453]
Epoch 66 completed. Loss: 0.1453
Best loss - saving model
Epoch 67/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.1477]
Epoch 67 completed. Loss: 0.1477
Epoch 68/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=0.1458]
Epoch 68 completed. Loss: 0.1458
Epoch 69/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=0.1323]
Epoch 69 completed. Loss: 0.1323
Best loss - saving model
Epoch 70/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1170]
Epoch 70 completed. Loss: 0.1170
Best loss - saving model
Epoch 71/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.1279]
Epoch 71 completed. Loss: 0.1279
Epoch 72/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.1379]
Epoch 72 completed. Loss: 0.1379
Epoch 73/90: 100%|██████████| 660/660 [01:46<00:00,  6.23batch/s, loss=0.1290]
Epoch 73 completed. Loss: 0.1290
Epoch 74/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1303]
Epoch 74 completed. Loss: 0.1303
Epoch 75/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.1102]
Epoch 75 completed. Loss: 0.1102
Best loss - saving model
Epoch 76/90: 100%|██████████| 660/660 [01:46<00:00,  6.23batch/s, loss=0.1183]
Epoch 76 completed. Loss: 0.1183
Epoch 77/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.1172]
Epoch 77 completed. Loss: 0.1172
Epoch 78/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.1124]
Epoch 78 completed. Loss: 0.1124
Epoch 79/90: 100%|██████████| 660/660 [01:45<00:00,  6.28batch/s, loss=0.1155]
Epoch 79 completed. Loss: 0.1155
Epoch 80/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.0985]
Epoch 80 completed. Loss: 0.0985
Best loss - saving model
Epoch 81/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.0989]
Epoch 81 completed. Loss: 0.0989
Epoch 82/90: 100%|██████████| 660/660 [01:46<00:00,  6.21batch/s, loss=0.0994]
Epoch 82 completed. Loss: 0.0994
Epoch 83/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.0814]
Epoch 83 completed. Loss: 0.0814
Best loss - saving model
Epoch 84/90: 100%|██████████| 660/660 [01:46<00:00,  6.23batch/s, loss=0.0964]
Epoch 84 completed. Loss: 0.0964
Epoch 85/90: 100%|██████████| 660/660 [01:46<00:00,  6.22batch/s, loss=0.0850]
Epoch 85 completed. Loss: 0.0850
Epoch 86/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.0882]
Epoch 86 completed. Loss: 0.0882
Epoch 87/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.0887]
Epoch 87 completed. Loss: 0.0887
Epoch 88/90: 100%|██████████| 660/660 [01:45<00:00,  6.23batch/s, loss=0.0983]
Epoch 88 completed. Loss: 0.0983
Epoch 89/90: 100%|██████████| 660/660 [01:46<00:00,  6.23batch/s, loss=0.0961]
Epoch 89 completed. Loss: 0.0961
Epoch 90/90: 100%|██████████| 660/660 [01:45<00:00,  6.24batch/s, loss=0.0845]
Epoch 90 completed. Loss: 0.0845
Training completed. Best loss: 0.0814


