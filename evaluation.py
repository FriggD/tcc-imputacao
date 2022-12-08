import torch
import random
import numpy as np

from tensorHelpers import tensorFromSentence
from device import device
from logger import logger

SOS_token = 0
EOS_token = 1

def evaluateRandomly(pairs, lang, encoder, decoder, args, n=0):
    for i in range(n):
        pair = random.choice(pairs)
        output_words, attentions = evaluate(lang, encoder, decoder, pair[0], args)
        output_sentence = ' '.join(output_words)
        logger.info(f'= {len(pair[1])} {pair[1]}')
        logger.info(f'< {len(output_sentence)} {output_sentence}')
        logger.info('')
        
def evaluateAll(modelData, lang, encoder, decoder, args, n=None, computeMafDist=False):
    acertos = 0
    tentativas = 0
    if n:
        pairs = random.sample(modelData.test, n)
    else:
        pairs = modelData.test
    
    markerMutations = np.zeros(len(modelData.mafArr))
    markerMutationPred = np.zeros(len(modelData.mafArr))
    
    for idx, pair in enumerate(pairs):
        real = np.array(list(pair[1].replace(" ", "")), dtype=int)
        output_words, attentions = evaluate(lang, encoder, decoder, pair[0], args)
        prediction = np.array(list(''.join(output_words).replace("<EOS>", "")), dtype=int)
        if len(prediction) < len(real):
            teste = np.full(len(real)-len(prediction), 2, dtype=int)
            prediction = np.concatenate((prediction, teste))
            
        tentativas += len(real)
        acertos += np.sum(real == prediction[:len(real)])
        if computeMafDist:
            for idx in range(len(real)):
                if real[idx] == 1:
                    markerMutations[idx] += 1 
                    if prediction[idx] == 1:
                        markerMutationPred[idx] += 1
    if computeMafDist:     
        common = [0,0,0]
        uncommon = [0,0,0]
        rare = [0,0,0]
        unheard = [0,0,0]
        for idx, maf in enumerate(modelData.mafArr):
            if maf > 0.05:
                common[0] += markerMutations[idx]
                common[1] += markerMutationPred[idx]
            elif maf > 0.005:
                uncommon[0] += markerMutations[idx]
                uncommon[1] += markerMutationPred[idx]
            elif maf > 0.000001:
                rare[0] += markerMutations[idx]
                rare[1] += markerMutationPred[idx]
            else:
                unheard[0] += markerMutations[idx]
                unheard[1] += markerMutationPred[idx]
        common[2] = 100*common[1]/common[0] if common[0] > 0 else 0
        uncommon[2] = 100*uncommon[1]/uncommon[0] if uncommon[0] > 0 else 0
        rare[2] = 100*rare[1]/rare[0] if rare[0] > 0 else 0
        
    logger.info(f"Common ({common[0]} {common[2]}%); uncommon ({uncommon[0]} {uncommon[2]}%); rare  ({rare[0]} {rare[2]}%);")
    
    return [
        acertos/tentativas, # Acurácia gerla
        common[0], common[1], common[2],
        uncommon[0], uncommon[1], uncommon[2],
        rare[0], rare[1], rare[2]
    ]
        
def evaluate(lang, encoder, decoder, sentence, args):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang[0], sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(args.max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(args.max_length, args.max_length)

        for di in range(args.max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang[1].index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]