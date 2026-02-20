import multiprocessing, torch, numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from train_functions import train_baseline, train_teacher_Encoder_Decoder, train_student_Encoder_Decoder
from error_grids import zone_accuracy
from ClarkeErrorGrid import clarke_error_grid
import matplotlib.pyplot as plt
import pandas as pd

def test_baseline_seq2seq(data, model_pop, file_name, args, category='clinical'):
    RMSEs = []
    MAEs = []
    # RMSEs_val = []
    # MAEs_val = []
    preds30 = []
    preds60 = []
    Y30 = []
    Y60 = []
    def compute_pred_error(pid):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_pop.load_state_dict(torch.load(file_name))
        model = train_baseline(model_pop, data[pid]['train'][:, :-1, args.variable_index], data[pid]['trainLabels'],
                               data[pid]['validate'][:, :-1, args.variable_index], data[pid]['validateLabels'],
                               lr=1e-5, warm_up=10, finetune=True, patients=40, seq2seq=args.return_sequence, epochs=500)
        X_test = torch.tensor(data[pid]['test'][:, :-1, args.variable_index], dtype=torch.float32).to(device)
        y_test = data[pid]['testLabels'] * 360.0
        y_test = y_test + np.repeat(np.reshape(data[pid]['test'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360+40.0
        # X_val = torch.tensor(data[pid]['validate'][:, :-1, args.variable_index], dtype=torch.float32).to(device)
        # y_val = data[pid]['validateLabels'] *360.0 + np.repeat(np.reshape(data[pid]['validate'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360.0 + 40.0
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(X_test).cpu().numpy() * 360.0
            pred = pred + np.repeat(np.reshape(data[pid]['test'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360+40.0
            pred[pred < 40.0] = 40.0
            pred[pred > 400.0] = 400.0
            mse = []
            mae = []
            # mse_val = []
            # mae_val = []
            # pred_val = model(X_val).cpu().numpy() * 360.0
            # pred_val = pred_val + np.repeat(np.reshape(data[pid]['validate'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360.0 + 40.0
            # pred_val[pred_val < 40.0] = 40.0
            # pred_val[pred_val > 400.0] = 400.0
            for ind in range(args.pred_horizon):
                mse.append(root_mean_squared_error(y_test[:, ind], pred[:, ind]))
                mae.append(mean_absolute_error(y_test[:, ind], pred[:, ind]))
                # mse_val.append(root_mean_squared_error(y_val[:, ind], pred_val[:, ind]))
                # mae_val.append(mean_absolute_error(y_val[:, ind], pred_val[:, ind]))
        return mse, mae#, y_test[:, 5], pred[:, 5], y_test[:, -1], pred[:, -1]
    if category == 'clinical':
        for pid in args.pid_list:
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
            # Y30.append(y1)
            # preds30.append(p1)
            # Y60.append(y2)
            # preds60.append(p2)
            # RMSEs_val.append(mse_val)
            # MAEs_val.append(mae_val)
    elif category == 'sim':
        for ind in range(1, 21):
            pid = str(ind)
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    else:
        raise ValueError('Invalid category')
    print(RMSEs[:, 5])
    print(RMSEs[:, 11])
    print(RMSEs[:, 17])
    print(RMSEs[:, 23])
    print(MAEs[:, 5])
    print(MAEs[:, 11])
    print(MAEs[:, 17])
    print(MAEs[:, 23])
    print(np.mean(np.array(RMSEs), axis=0), np.std(np.array(RMSEs), axis=0))
    print(np.mean(np.array(MAEs), axis=0), np.std(np.array(MAEs), axis=0))


def test_seq2seq_teacher(data, model_pop_enc, model_pop_dec, file_name_enc, file_name_dec, args, category='clinical'):
    RMSEs = []
    MAEs = []
    def compute_pred_error(pid):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_pop_enc.load_state_dict(torch.load(file_name_enc))
        model_pop_dec.load_state_dict(torch.load(file_name_dec))
        model_enc, model_dec = train_teacher_Encoder_Decoder(model_pop_enc, model_pop_dec, data[pid]['train'][:, :-1, args.variable_index], data[pid]['trainLabels'],
                               data[pid]['validate'][:, :-1, args.variable_index], data[pid]['validateLabels'],
                               lr=1e-5, warm_up=10, finetune=True, patients=40, seq2seq=args.return_sequence, epochs=500) ## 1e-5 for all other models
        torch.save(model_enc.state_dict(), 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_Encoder'+ pid +'.pt')
        torch.save(model_dec.state_dict(), 'AZT1DM_T1DM_Transformer_Seq2seq_Teacher_Decoder'+ pid +'.pt')
        X_test = torch.tensor(data[pid]['test'][:, :-1, args.variable_index], dtype=torch.float32).to(device)
        y_test = data[pid]['testLabels'] * 360.0
        y_test = y_test + np.repeat(np.reshape(data[pid]['test'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360+40.0
        model_enc.to(device)
        model_dec.to(device)
        model_enc.eval()
        model_dec.eval()
        with (torch.no_grad()):
            mem = model_enc(X_test)
            pred = model_dec(mem).cpu().numpy() * 360.0
            pred = pred + np.repeat(np.reshape(data[pid]['test'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360+40.0
            pred[pred < 40.0] = 40.0
            pred[pred > 400.0] = 400.0
            mse = []
            mae = []
            for ind in range(args.pred_horizon):
                mse.append(root_mean_squared_error(y_test[:, ind], pred[:, ind]))
                mae.append(mean_absolute_error(y_test[:, ind], pred[:, ind]))
        return mse, mae
    if category == 'clinical':
        for pid in args.pid_list:
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    elif category == 'sim':
        for ind in range(1, 21):
            pid = str(ind)
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    else:
        raise ValueError('Invalid category')
    print(np.mean(np.array(RMSEs), axis=0), np.std(np.array(RMSEs), axis=0))
    print(np.mean(np.array(MAEs), axis=0), np.std(np.array(MAEs), axis=0))



def test_seq2seq_student(data, teacher_pop_enc, teacher_pop_dec, student_pop_enc, student_pop_dec,
                         file_name_teacher_enc, file_name_teacher_dec, file_name_student_enc, file_name_student_dec,
                         args, category='clinical'):
    RMSEs = []
    MAEs = []
    preds30 = []
    preds60 = []
    preds90 = []
    preds120 = []
    Y30 = []
    Y60 = []
    Y90 = []
    Y120 = []
    def compute_pred_error(pid):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher_pop_enc.load_state_dict(torch.load(file_name_teacher_enc+pid+'.pt'))
        teacher_pop_dec.load_state_dict(torch.load(file_name_teacher_dec+pid+'.pt'))
        student_pop_enc.load_state_dict(torch.load(file_name_student_enc+'.pt'))
        student_pop_dec.load_state_dict(torch.load(file_name_student_dec+'.pt'))
        student_pop_dec.load_state_dict(torch.load(file_name_teacher_dec+pid+'.pt'))
        student_enc, student_dec = train_student_Encoder_Decoder(teacher_pop_enc, teacher_pop_dec, student_pop_enc, student_pop_dec,
                                                                 data[pid]['train'][:, :-1, args.variable_index],
                                                                 data[pid]['trainLabels'],
                                                                 data[pid]['validate'][:, :-1, args.variable_index],
                                                                 data[pid]['validateLabels'], args,
                                                                 lr=1e-4, warm_up=10, patients=40,
                                                                 seq2seq=args.return_sequence)
        model_enc, model_dec = train_student_Encoder_Decoder(teacher_pop_enc, teacher_pop_dec, student_enc, student_dec, data[pid]['train'][:, :-1, args.variable_index], data[pid]['trainLabels'],
                               data[pid]['validate'][:, :-1, args.variable_index], data[pid]['validateLabels'], args,
                               lr=1e-4, warm_up=10, finetune=True, patients=60, seq2seq=args.return_sequence, epochs=500) ## 1e-5 for all other models
        torch.save(model_enc.state_dict(), file_name_student_enc + pid +'.pt')
        torch.save(model_dec.state_dict(), file_name_student_dec+ pid +'.pt')
        # model_enc = student_pop_enc
        # model_dec = teacher_pop_dec
        # model_enc.load_state_dict(torch.load(file_name_student_enc+pid+'.pt'))
        # model_dec.load_state_dict(torch.load(file_name_student_dec+pid+'.pt'))
        X_test = torch.tensor(data[pid]['test'][:, :args.seq_len, args.variable_index], dtype=torch.float32).to(device)
        y_test = data[pid]['testLabels'] * 360.0
        y_test = y_test + np.repeat(np.reshape(data[pid]['test'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360+40.0

        model_enc.to(device)
        model_dec.to(device)
        model_enc.eval()
        model_dec.eval()
        with (torch.no_grad()):
            mem = model_enc(X_test)
            pred = model_dec(mem).cpu().numpy() * 360.0
            pred = pred + np.repeat(np.reshape(data[pid]['test'][:, args.seq_len-1, 0], shape=(-1, 1)), args.pred_horizon, axis=1)*360+40.0
            pred[pred < 40.0] = 40.0
            pred[pred > 400.0] = 400.0
            mse = []
            mae = []
            for ind in range(args.pred_horizon):
                mse.append(root_mean_squared_error(y_test[:, ind], pred[:, ind]))
                mae.append(mean_absolute_error(y_test[:, ind], pred[:, ind]))
                # if pid =='588':
                #     plt.figure(figsize=(8,6))
                #     plt.fill([data[pid]['testTimes'][0], data[pid]['testTimes'][-1], data[pid]['testTimes'][-1], data[pid]['testTimes'][0]],
                #              [0, 0, 70, 70], color='orange', alpha=0.3, label='Hypoglycemia')
                #     plt.fill([data[pid]['testTimes'][0], data[pid]['testTimes'][-1], data[pid]['testTimes'][-1], data[pid]['testTimes'][0]],
                #              [180, 180, 400, 400], color='red', alpha=0.2, label='Hyperglycemia')
                #     plt.plot(data[pid]['testTimes'], y_test[:, ind], 'k', label='Measured')
                #     plt.plot(data[pid]['testTimes'], pred[:, ind], 'b--', label='Predicted')
                #     plt.xlabel('Time')
                #     plt.ylabel('BGL (mg/dL)')
                #     plt.xlim(data[pid]['testTimes'][950], data[pid]['testTimes'][1600])
                #     plt.ylim(50, 320)
                #     plt.legend(loc='upper left')
                #     plt.title(pid+': '+str(ind+1))
                #     plt.show()
        data_frame1 = pd.DataFrame(y_test)
        data_frame2 = pd.DataFrame(pred)
        data_frame1.to_csv('./results/'+pid +'_y_test.csv', index=False)
        data_frame2.to_csv('./results/'+pid +'_y_pred.csv', index=False)
        return mse, mae, y_test[:, 5], pred[:, 5], y_test[:, 11], pred[:, 11], y_test[:, 17], pred[:, 17], y_test[:, 23], pred[:, 23]
    if category == 'clinical':
        for pid in args.pid_list:
            mse, mae, y1, p1, y2, p2, y3, p3, y4, p4 = compute_pred_error(pid)
            Y30.append(y1)
            preds30.append(p1)
            Y60.append(y2)
            preds60.append(p2)
            Y90.append(y3)
            preds90.append(p3)
            Y120.append(y4)
            preds120.append(p4)
            RMSEs.append(mse)
            MAEs.append(mae)
    elif category == 'sim':
        for ind in range(1, 21):
            pid = str(ind)
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    else:
        raise ValueError('Invalid category')
    # Y30 = np.concatenate(Y30)
    # Y60 = np.concatenate(Y60)
    # preds30 = np.concatenate(preds30)
    # preds60 = np.concatenate(preds60)
    # Y90 = np.concatenate(Y90)
    # Y120 = np.concatenate(Y120)
    # preds90 = np.concatenate(preds90)
    # preds120 = np.concatenate(preds120)
    # acc1 = zone_accuracy(Y30, preds30)
    # acc2 = zone_accuracy(Y60, preds60)
    # clarke_error_grid(Y30, preds30, title="Clarke Error Grid Analysis")
    # clarke_error_grid(Y60, preds60, title="Clarke Error Grid Analysis")
    # clarke_error_grid(Y90, preds90, title="Clarke Error Grid Analysis")
    # clarke_error_grid(Y120, preds120, title="Clarke Error Grid Analysis")
    # print(acc1, acc2)
    print(np.mean(np.array(RMSEs), axis=0), np.std(np.array(RMSEs), axis=0))
    print(np.mean(np.array(MAEs), axis=0), np.std(np.array(MAEs), axis=0))
    # print(np.mean(np.array(RMSEs[:6]), axis=0), np.std(np.array(RMSEs[:6]), axis=0))
    # print(np.mean(np.array(MAEs[:6]), axis=0), np.std(np.array(MAEs[:6]), axis=0))
    # print(np.mean(np.array(RMSEs[6:]), axis=0), np.std(np.array(RMSEs[6:]), axis=0))
    # print(np.mean(np.array(MAEs[6:]), axis=0), np.std(np.array(MAEs[6:]), axis=0))


def test_baseline_OneStep(data, model_pop, file_name, args, category='clinical'):
    RMSEs = []
    MAEs = []
    RMSEs_Val = []
    MAEs_Val = []
    def compute_pred_error(pid):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_pop.load_state_dict(torch.load(file_name))
        model = train_baseline(model_pop, data[pid]['train'][:, :-1, args.variable_index], data[pid]['trainLabels'],
                               data[pid]['validate'][:, :-1, args.variable_index], data[pid]['validateLabels'],
                               lr=1e-5, warm_up=10, finetune=True, patients=60)
        model.to(device)
        model.eval()
        X_test = torch.tensor(data[pid]['test'][:, :-1, args.variable_index], dtype=torch.float32).to(
            device)
        y_test = data[pid]['testLabels'] * 360.0
        y_test = y_test + data[pid]['test'][:, -2, 0] * 360 + 40
        with torch.no_grad():
            pred = model(X_test).cpu().numpy() * 360.0
            pred = pred + data[pid]['test'][:, -2, 0] * 360 + 40
            pred[pred < 40.0] = 40.0
            pred[pred > 400.0] = 400.0
            mse = root_mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
        return mse, mae

    if category == 'clinical':
        for pid in args.pid_list:
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    elif category == 'sim':
        for ind in range(1, 21):
            pid = str(ind)
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    else:
        raise ValueError('Invalid category')
    # print(np.array(RMSEs), np.array(RMSEs))
    # print(np.array(MAEs), np.array(MAEs))
    print(np.mean(np.array(RMSEs), axis=0), np.std(np.array(RMSEs), axis=0))
    print(np.mean(np.array(MAEs), axis=0), np.std(np.array(MAEs), axis=0))
    print(np.mean(np.array(RMSEs[:6]), axis=0), np.std(np.array(RMSEs[:6]), axis=0))
    print(np.mean(np.array(MAEs[:6]), axis=0), np.std(np.array(MAEs[:6]), axis=0))
    print(np.mean(np.array(RMSEs[6:]), axis=0), np.std(np.array(RMSEs[6:]), axis=0))
    print(np.mean(np.array(MAEs[6:]), axis=0), np.std(np.array(MAEs[6:]), axis=0))


def test_teacher_model(data, teacher_model, decoder_model, args, category='clinical'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    decoder_model.to(device)
    teacher_model.eval()
    decoder_model.eval()
    RMSEs = []
    MAEs = []
    def compute_pred_error(pid):
        X_test = data[pid]['test'][:, :-1, args.variable_index]
        X_test[:, args.seq_len:, 0] = 0.0
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = data[pid]['test'][:, args.seq_len:-1, 0] * 360.0
        with (((torch.no_grad()))):
            mem = teacher_model(X_test)
            pred = decoder_model(mem).cpu().numpy() * 360.0
            # pred[pred<40.0] = 40.0
            # pred[pred>400.0] = 400.0
            mse = []
            mae = []
            for ind in range(args.pred_horizon):
                mse.append(root_mean_squared_error(y_test[:, ind], pred[:, ind]))
                mae.append(mean_absolute_error(y_test[:, ind], pred[:, ind]))
        return mse, mae
    if category == 'clinical':
        for pid in args.pid_list:
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    elif category == 'sim':
        for ind in range(1, 21):
            pid = str(ind)
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    else:
        raise ValueError('Invalid category')
    print(np.mean(np.array(RMSEs), axis=0), np.std(np.array(RMSEs), axis=0))
    print(np.mean(np.array(MAEs), axis=0), np.std(np.array(MAEs), axis=0))



def test_student_model(data, teacher_model, decoder_model, args, category='clinical'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    decoder_model.to(device)
    teacher_model.eval()
    decoder_model.eval()
    RMSEs = []
    MAEs = []
    def compute_pred_error(pid):
        X_test = data[pid]['test'][:, :args.seq_len, args.variable_index]
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = data[pid]['test'][:, args.seq_len:-1, 0] * 360.0
        with (((torch.no_grad()))):
            mem = teacher_model(X_test)
            pred = decoder_model(mem).cpu().numpy() * 360.0
            pred[pred<40.0] = 40.0
            pred[pred>400.0] = 400.0
            mse = []
            mae = []
            for ind in range(args.pred_horizon):
                mse.append(root_mean_squared_error(y_test[:, ind], pred[:, ind]))
                mae.append(mean_absolute_error(y_test[:, ind], pred[:, ind]))
        return mse, mae
    if category == 'clinical':
        for pid in args.pid_list:
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    elif category == 'sim':
        for ind in range(1, 21):
            pid = str(ind)
            mse, mae = compute_pred_error(pid)
            RMSEs.append(mse)
            MAEs.append(mae)
    else:
        raise ValueError('Invalid category')
    print(np.mean(np.array(RMSEs), axis=0), np.std(np.array(RMSEs), axis=0))
    print(np.mean(np.array(MAEs), axis=0), np.std(np.array(MAEs), axis=0))