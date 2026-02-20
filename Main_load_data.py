import pickle, argparse, random, torch, numpy as np
from init_parameters import init_para
from load_data_from_file import load_Ohio_data
from process_data import prepare4learn
from baseline_models import *
from train_functions import *
from test_functions import *
from basic_transformer_models import *
from torchinfo import summary


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


####################################### 3 0  ############################################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 6
    with open('OhioT1DM_processed.pkl', 'rb') as f:
        _, data, times, isReal = pickle.load(f)
    f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    data, args = prepare4learn(args, data, times, isReal)
    model = LSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_LSTM_OneStep30.pt'
    model = BiLSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_BiLSTM_OneStep30.pt'
    torch.save(model.state_dict(), file_name)
    model = CRNN_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, patients=40)
    file_name = 'AZT1DM_T1DM_CRNN_OneStep30.pt'
    torch.save(model.state_dict(), file_name)
    model = Transformer_baseline_model(map_dim=6, seq_len=args.seq_len, pred_horizon=1, d_model=64, num_heads=4, num_enc_layer=3, d_ff=128)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_Transformer_OneStep30.pt'
    torch.save(model.state_dict(), file_name)
    test_baseline_OneStep(data, model, file_name, args, category='clinical')
    summary(model)
    ####################################### 6 0  ############################################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 12
    with open('OhioT1DM_processed.pkl', 'rb') as f:
        _, data, times, isReal = pickle.load(f)
    f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    data, args = prepare4learn(args, data, times, isReal)
    model = LSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_LSTM_OneStep60.pt'
    torch.save(model.state_dict(), file_name)
    model = BiLSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_BiLSTM_OneStep60.pt'
    torch.save(model.state_dict(), file_name)
    model = CRNN_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_CRNN_OneStep60.pt'
    torch.save(model.state_dict(), file_name)
    model = Transformer_baseline_model(map_dim=6, seq_len=args.seq_len, pred_horizon=1, d_model=64, num_heads=4,
                                       num_enc_layer=3, d_ff=128)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_Transformer_OneStep60.pt'
    torch.save(model.state_dict(), file_name)
    test_baseline_OneStep(data, model, file_name, args, category='clinical')
####################################### 9 0  ############################################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 18
    with open('OhioT1DM_processed.pkl', 'rb') as f:
        _, data, times, isReal = pickle.load(f)
    f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    data, args = prepare4learn(args, data, times, isReal)
    model = LSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_LSTM_OneStep90.pt'
    torch.save(model.state_dict(), file_name)
    model = BiLSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_BiLSTM_OneStep90.pt'
    torch.save(model.state_dict(), file_name)
    model = CRNN_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_CRNN_OneStep90.pt'
    torch.save(model.state_dict(), file_name)
    model = Transformer_baseline_model(map_dim=6, seq_len=args.seq_len, pred_horizon=1, d_model=64, num_heads=4,
                                       num_enc_layer=3, d_ff=128)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_Transformer_OneStep90.pt'
    torch.save(model.state_dict(), file_name)
    test_baseline_OneStep(data, model, file_name, args, category='clinical')
####################################### 1 2 0  ############################################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 24
    with open('OhioT1DM_processed.pkl', 'rb') as f:
        _, data, times, isReal = pickle.load(f)
    f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    data, args = prepare4learn(args, data, times, isReal)
    model = LSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10)
    file_name = 'AZT1DM_T1DM_LSTM_OneStep120.pt'
    torch.save(model.state_dict(), file_name)
    model = BiLSTM_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10)
    file_name = 'AZT1DM_T1DM_BiLSTM_OneStep120.pt'
    torch.save(model.state_dict(), file_name)
    model = CRNN_model(channels=len(args.variable_index))
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_CRNN_OneStep120.pt'
    torch.save(model.state_dict(), file_name)
    model = Transformer_baseline_model(map_dim=6, seq_len=args.seq_len, pred_horizon=1, d_model=64, num_heads=4,
                                       num_enc_layer=3, d_ff=128)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'], lr=1e-4, warm_up=10, patients=40)
    file_name = 'AZT1DM_T1DM_Transformer_OneStep120.pt'
    torch.save(model.state_dict(), file_name)
    test_baseline_OneStep(data, model, file_name, args, category='clinical')
######################################### Seq2seq upto 1 2 0 min #####################################################################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 24
    args.return_sequence = True
    with open('OhioT1DM_processed.pkl', 'rb') as f:
        _, data, times, isReal = pickle.load(f)
    f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    args.pid_list = [str(i) for i in range(1, 26)]
    data, args = prepare4learn(args, data, times, isReal)
    model = LSTMSeq2seq(channels=len(args.variable_index), pred_horizon=args.pred_horizon)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'],
                           lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence)
    file_name = 'AZT1DM_T1DM_LSTM_Seq2seq.pt'
    torch.save(model.state_dict(), file_name)
    model = BiLSTMSeq2seq(channels=len(args.variable_index), pred_horizon=args.pred_horizon)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'],
                           lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence)
    file_name = 'AZT1DM_T1DM_BiLSTM_Seq2seq.pt'
    torch.save(model.state_dict(), file_name)
    model = CRNNSeq2seq(channels=len(args.variable_index), pred_horizon=args.pred_horizon, seq_len=args.seq_len)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'],
                           lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence)
    file_name = 'AZT1DM_T1DM_CRNN_Seq2seq.pt'
    torch.save(model.state_dict(), file_name)
    model = Transformer_seq2seq_model(channels=len(args.variable_index), pred_horizon=args.pred_horizon, d_model=64, num_heads=4,
                                       num_enc_layer=3, d_ff=128)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'],
                           lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence)
    file_name = 'AZT1DM_T1DM_Transformer_Seq2seqt.pt'
    torch.save(model.state_dict(), file_name)
    test_baseline_seq2seq(data, model, file_name, args, category='clinical')
    print(args)
################################### Teachers with future disturbances ########################################################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 24
    args.return_sequence = True
    args.keep_future = True
    # with open('OhioT1DM_processed.pkl', 'rb') as f:
    #     _, data, times, isReal = pickle.load(f)
    # f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    args.pid_list = [str(i) for i in range(1, 26)]
    data, args = prepare4learn(args, data, times, isReal)
    model = Transformer_seq2seq_Teacher_model(channels=len(args.variable_index), pred_horizon=args.pred_horizon,
                                              d_model=64, num_heads=4,
                                              num_enc_layer=3, d_ff=128)
    summary(model)
    model = train_baseline(model, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'],
                           lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence)
    file_name = 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_baseline.pt'
    torch.save(model.state_dict(), file_name)
    test_baseline_seq2seq(data, model, file_name, args, category='clinical')
################################## Train Teacher-Encoder-Decoder with future disturbances ######################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 24
    args.return_sequence = True
    args.keep_future = True
    # with open('OhioT1DM_processed.pkl', 'rb') as f:
    #     _, data, times, isReal = pickle.load(f)
    # f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    args.pid_list = [str(i) for i in range(1, 26)]
    data, args = prepare4learn(args, data, times, isReal)
    model_enc = Transformer_seq2seq_Encoder_model(channels=len(args.variable_index), pred_horizon=args.pred_horizon,
                                              d_model=64, num_heads=4,
                                              num_enc_layer=3, d_ff=128)
    model_dec = Transformer_seq2seq_Decoder_model(pred_horizon=args.pred_horizon, d_model=64)
    summary(model_enc)
    summary(model_dec)
    model_enc, model_dec = train_teacher_Encoder_Decoder(model_enc, model_dec, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                           data['validate'][:, :-1, args.variable_index], data['validateLabels'],
                           lr=1e-4, warm_up=10, patients=10, seq2seq=args.return_sequence)
    file_name_enc = 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_Encoder.pt'
    file_name_dec = 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_Decoder.pt'
    torch.save(model_enc.state_dict(), file_name_enc)
    torch.save(model_dec.state_dict(), file_name_dec)
    test_seq2seq_teacher(data, model_enc, model_dec, file_name_enc, file_name_dec, args, category='clinical')
    print(args)
################################## Train KD with future disturbances ######################
    args = init_para(argparse.ArgumentParser())
    args.pred_horizon = 24
    args.return_sequence = True
    args.keep_future = True
    with open('OhioT1DM_processed.pkl', 'rb') as f:
        _, data, times, isReal = pickle.load(f)
    f.close()
    with open('AZT1DM.pkl', 'rb') as f:
        data, times, isReal = pickle.load(f)
    f.close()
    args.pid_list = [str(i) for i in range(1, 26)]
    data, args = prepare4learn(args, data, times, isReal)
    teacher_enc = Transformer_seq2seq_Encoder_model(channels=len(args.variable_index), pred_horizon=args.pred_horizon,
                                              d_model=64, num_heads=4,
                                              num_enc_layer=3, d_ff=128)
    teacher_dec = Transformer_seq2seq_Decoder_model(pred_horizon=args.pred_horizon, d_model=64)
    student_enc = Transformer_seq2seq_Student_Encoder_model(channels=len(args.variable_index), seq_len=args.seq_len, pred_horizon=args.pred_horizon,
                                              d_model=64, num_heads=4,
                                              num_enc_layer=3, d_ff=128)
    student_dec = Transformer_seq2seq_Decoder_model(pred_horizon=args.pred_horizon, d_model=64)
    file_name_teacher_enc = 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_Encoder'
    file_name_teacher_dec = 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_Decoder'
    teacher_enc.load_state_dict(torch.load(file_name_teacher_enc+'.pt'))
    teacher_dec.load_state_dict(torch.load(file_name_teacher_dec+'.pt'))
    student_dec.load_state_dict(torch.load(file_name_teacher_dec+'.pt'))
    student_enc, student_dec = train_teacher_Encoder_Decoder(student_enc, student_dec, data['train'][:, :args.seq_len, args.variable_index], data['trainLabels'],
                                                             data['validate'][:, :args.seq_len, args.variable_index], data['validateLabels'],
                                                             lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence, train_decocer=False)
    torch.save(student_enc.state_dict(), 'AZT1DMStudent_Enc_baseline.pt')
    torch.save(student_dec.state_dict(), 'AZT1DMStudent_Dec_baseline.pt')
    student_enc.load_state_dict(torch.load('AZT1DMStudent_Enc_baseline.pt'))
    student_dec.load_state_dict(torch.load('AZT1DMStudent_Dec_baseline.pt'))

    student_enc, student_dec = train_student_Encoder_Decoder(teacher_enc, teacher_dec, student_enc, student_dec, data['train'][:, :-1, args.variable_index], data['trainLabels'],
                                                             data['validate'][:, :-1, args.variable_index], data['validateLabels'], args,
                                                             lr=1e-4, warm_up=10, patients=40, seq2seq=args.return_sequence)
    file_name_student_enc = 'AZT1DM_T1DM_Transformer_Seq2seq_Student_Encoder'
    file_name_student_dec = 'AZT1DM_T1DM_Transformer_Seq2seq_Student_Decoder'
    torch.save(student_enc.state_dict(), file_name_student_enc+'.pt')
    torch.save(student_dec.state_dict(), file_name_student_dec+'.pt')
    test_seq2seq_student(data, teacher_enc, teacher_dec, student_enc, student_dec, file_name_teacher_enc, file_name_teacher_dec, file_name_student_enc, file_name_student_dec, args, category='clinical')
    print(args)