import torch
import random
import time
import pandas as pd

from helpers import timeSince
from plot import showPlot
from evaluation import evaluate, evaluateRandomly, evaluateAll
from tensorHelpers import tensorsFromPair, tensorFromSentence

from logger import logger
from device import device

SOS_token = 0
EOS_token = 1

  
def trainIters(lang, modelData, encoder, decoder, args):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every args.eval_ev
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lrate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=args.lrate)
    training_pairs = [tensorsFromPair(lang, random.choice(modelData.train)) for i in range(args.it)]
    criterion = torch.nn.NLLLoss()

    accuracies = []
    for iter in range(1, args.it + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, args)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % args.eval_ev == 0:
            acc = evaluateAll(modelData, lang, encoder, decoder, args, n=100, computeMafDist=True)
            accuracies.append(acc)
            
            print_loss_avg = print_loss_total / args.eval_ev
            print_loss_total = 0
            logger.info('%s (%d %d%%) LOSS %.4f ACC %.4f'  % (timeSince(start, iter / args.it), iter, (iter / args.it * 100), print_loss_avg, acc[0]))
            if args.verbose > 1:
                evaluateRandomly(modelData.test, lang, encoder, decoder, args, n=1)
    
      
    acc = evaluateAll(modelData, lang, encoder, decoder, args, computeMafDist=True)
    accuracies.append(acc)
    accPD = pd.DataFrame(accuracies,columns=["Acc", "common positives", "common acertos", "common acc","uncommon positives", "uncommon acertos", "uncommon acc","rare positives", "rare acertos", "rare acc"])
    accPD.to_csv("log/Acc_"+str(int(time.time()))+".csv")
  
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(args.max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length