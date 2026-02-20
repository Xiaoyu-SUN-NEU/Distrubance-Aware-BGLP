import torch, numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F


def train_baseline(model, train_in, train_out, val_in, val_out, lr=1e-4, batch_size=512,
                   epochs=500, patients=20, warm_up=10, finetune=False, seq2seq=False):
    def warm_up_lambda(epoch):
        return float(epoch + 1) / float(warm_up)
    ##################### initialize optimization ################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).to(torch.float32)
    train_in = torch.tensor(train_in, dtype=torch.float32).to(device)
    train_out = torch.tensor(train_out, dtype=torch.float32).to(device)
    val_in = torch.tensor(val_in, dtype=torch.float32).to(device)
    val_out = torch.tensor(val_out, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warm_up, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_lambda)
    if seq2seq:
        # out_std = torch.std(train_out, dim=0)
        median = train_out.median(dim=0).values
        mad = (train_out - median.unsqueeze(0)).abs().median(dim=0).values
        out_std = 1.4826 * mad
        criterion = StdweightedMSELoss(out_std)
    else:
        criterion = nn.MSELoss()
    counter = 0
    best_val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_in), 4096):
            pred = model(val_in[i:i + 4096])
            if seq2seq:
                best_val_loss += criterion(pred, val_out[i:i + 4096]).item()
            else:
                best_val_loss += mean_squared_error(pred.cpu().numpy(), val_out[i:i + 4096].cpu().numpy())
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(0, len(train_in), batch_size):
            temp_in = train_in[i:i+batch_size]
            temp_out = train_out[i:i+batch_size]
            if len(temp_in) < batch_size:
                temp_in = train_in[-batch_size:]
                temp_out = train_out[-batch_size:]
            pred = model(temp_in)
            optimizer.zero_grad()
            loss = criterion(pred, temp_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(temp_in)
        epoch_loss /= len(train_in)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_in), 4096):
                pred = model(val_in[i:i+4096])
                if seq2seq:
                    val_loss += criterion(pred, val_out[i:i+4096]).item()
                else:
                    val_loss += mean_squared_error(pred.cpu().numpy(), val_out[i:i+4096].cpu().numpy())
        print(
            f'Epoch [{epoch + 1}/{epochs}], training loss:{epoch_loss:.6f}, val loss: {val_loss:.6f}, counter: {counter}')
        # scheduler.step(val_loss)
        if epoch < warm_up:
            warmup_scheduler.step()
        else:
            scheduler.step()
            # scheduler.step(val_loss)
        if (val_loss < best_val_loss):  # save model with lowest loss
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved")
        else:
            counter += 1
        if counter >= patients:  # early stop
            print(f'Early stopping triggerd after {epoch + 1} epochs. Validation loss: {best_val_loss:.6f}')
            model.load_state_dict(torch.load('best_model.pth'))
            print("Best model loaded after early stopping")
            break
    return model


def train_teacher_Encoder_Decoder(model_enc, model_dec, train_in, train_out, val_in, val_out, lr=1e-4, batch_size=512,
                   epochs=500, patients=20, warm_up=10, finetune=False, seq2seq=False, train_decocer=True):
    def warm_up_lambda(epoch):
        return float(epoch + 1) / float(warm_up)
    ##################### initialize optimization ################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_enc.to(device).to(torch.float32)
    model_dec.to(device).to(torch.float32)
    if train_decocer is False:
        for param in model_dec.parameters():
            param.requires_grad = False
    train_in = torch.tensor(train_in, dtype=torch.float32).to(device)
    train_out = torch.tensor(train_out, dtype=torch.float32).to(device)
    val_in = torch.tensor(val_in, dtype=torch.float32).to(device)
    val_out = torch.tensor(val_out, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(list(model_enc.parameters())+list(model_dec.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warm_up, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_lambda)
    if seq2seq:
        # out_std = torch.std(train_out, dim=0)
        median = train_out.median(dim=0).values
        mad = (train_out - median.unsqueeze(0)).abs().median(dim=0).values
        out_std = 1.4826 * mad
        criterion = StdweightedMSELoss(out_std)
    else:
        criterion = nn.MSELoss()
    counter = 0
    best_val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_in), 4096):
            mem = model_enc(val_in[i:i + 4096])
            pred = model_dec(mem)
            if seq2seq:
                best_val_loss += criterion(pred, val_out[i:i + 4096]).item()
            else:
                best_val_loss += mean_squared_error(pred.cpu().numpy(), val_out[i:i + 4096].cpu().numpy())
    for epoch in range(epochs):
        model_enc.train()
        model_dec.train()
        epoch_loss = 0.0
        for i in range(0, len(train_in), batch_size):
            temp_in = train_in[i:i+batch_size]
            temp_out = train_out[i:i+batch_size]
            if len(temp_in) < batch_size:
                temp_in = train_in[-batch_size:]
                temp_out = train_out[-batch_size:]
            mem = model_enc(temp_in)
            pred = model_dec(mem)
            optimizer.zero_grad()
            loss = criterion(pred, temp_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(temp_in)
        epoch_loss /= len(train_in)

        val_loss = 0.0
        model_enc.eval()
        model_dec.eval()
        with torch.no_grad():
            for i in range(0, len(val_in), 4096):
                mem = model_enc(val_in[i:i+4096])
                pred = model_dec(mem)
                if seq2seq:
                    val_loss += criterion(pred, val_out[i:i+4096]).item()
                else:
                    val_loss += mean_squared_error(pred.cpu().numpy(), val_out[i:i+4096].cpu().numpy())
        print(
            f'Epoch [{epoch + 1}/{epochs}], training loss:{epoch_loss:.6f}, val loss: {val_loss:.6f}, counter: {counter}')
        # scheduler.step(val_loss)
        if epoch < warm_up:
            warmup_scheduler.step()
        else:
            scheduler.step()
            # scheduler.step(val_loss)
        if (val_loss < best_val_loss):  # save model with lowest loss
            best_val_loss = val_loss
            counter = 0
            torch.save(model_enc.state_dict(), 'best_enc_model.pth')
            torch.save(model_dec.state_dict(), 'best_dec_model.pth')
            print("Best model saved")
        else:
            counter += 1
        if counter >= patients:  # early stop
            print(f'Early stopping triggerd after {epoch + 1} epochs. Validation loss: {best_val_loss:.6f}')
            model_enc.load_state_dict(torch.load('best_enc_model.pth'))
            model_dec.load_state_dict(torch.load('best_dec_model.pth'))
            print("Best model loaded after early stopping")
            break
    return model_enc, model_dec

def train_teacher(teacher_encoder, shared_decoder, train_in, train_out, val_in, val_out,
                  lr=1e-4, epochs=8000, batch_size=512, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    teacher_encoder.to(device).to(torch.float32)
    shared_decoder.to(device).to(torch.float32)
    train_in = torch.tensor(train_in, dtype=torch.float32).to(device)
    train_out = torch.tensor(train_out, dtype=torch.float32).to(device)
    val_in = torch.tensor(val_in, dtype=torch.float32).to(device)
    val_out = torch.tensor(val_out, dtype=torch.float32).to(device)
    # setup training profiles
    criterion = nn.MSELoss()
    optimizer_teacher = torch.optim.Adam(list(teacher_encoder.parameters()) + list(shared_decoder.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_teacher, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    # Train population model
    counter = 0
    for epoch in range(epochs):
        temp_ind = torch.randperm(len(train_in)).to(device)
        train_in = train_in[temp_ind]
        train_out = train_out[temp_ind]
        teacher_encoder.train()
        shared_decoder.train()
        epoch_loss = 0
        for i in range(0, len(train_in), batch_size):
            temp_in = train_in[i:i+batch_size]
            temp_out = train_out[i:i+batch_size]
            if len(temp_in) < batch_size:
                continue
            optimizer_teacher.zero_grad()
            teacher_mem = teacher_encoder(temp_in)

            teacher_output = shared_decoder(teacher_mem)  # , mem_mask=mem_mask)
            loss = criterion(teacher_output, temp_out)
            loss.backward()
            optimizer_teacher.step()
            epoch_loss += loss.item()
        teacher_encoder.eval()
        shared_decoder.eval()
        val_loss = 0
        with torch.no_grad():  # validation
            for i in range(0, len(val_in), 4096):
                teacher_mem = teacher_encoder(val_in[i:i + 4096])
                outputs = shared_decoder(teacher_mem)  #
                val_loss += mean_squared_error(outputs.cpu().numpy(), val_out[i:i + 4096].cpu().numpy())
        print(
            f'Epoch [{epoch + 1}/{epochs}], training loss:{epoch_loss:.6f}, val loss: {val_loss:.6f}, counter: {counter}')
        if epoch >= 10:
            if val_loss < best_val_loss:  # save model with lowest loss
                best_val_loss = val_loss
                counter = 0
                torch.save(teacher_encoder.state_dict(), 'best_teacher_model.pth')
                torch.save(shared_decoder.state_dict(), 'best_decoder_model.pth')
                print("Best model saved")
            else:
                counter += 1
            if counter >= patience:  # early stop
                print(f'Early stopping triggerd after {epoch + 1} epochs. Validation loss: {best_val_loss:.6f}')
                teacher_encoder.load_state_dict(torch.load('best_teacher_model.pth'))
                shared_decoder.load_state_dict(torch.load('best_decoder_model.pth'))
                print("Best model loaded after early stopping")
                break
            scheduler.step(val_loss)
    return teacher_encoder, shared_decoder



def train_student(teacher_encoder, teacher_decoder, student_encoder, student_decoder, train_in1, train_in2, train_out, val_in1, val_in2, val_out,
                  lr=1e-4, epochs=8000, batch_size=512, patience=20, train_decoder=False, align_loss_weight=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    teacher_encoder.to(device).to(torch.float32)
    teacher_decoder.to(device).to(torch.float32)
    student_encoder.to(device).to(torch.float32)
    student_decoder.to(device).to(torch.float32)
    train_in1 = torch.tensor(train_in1, dtype=torch.float32).to(device)
    train_in2 = torch.tensor(train_in2, dtype=torch.float32).to(device)
    train_out = torch.tensor(train_out, dtype=torch.float32).to(device)
    val_in1 = torch.tensor(val_in1, dtype=torch.float32).to(device)
    val_in2 = torch.tensor(val_in2, dtype=torch.float32).to(device)
    val_out = torch.tensor(val_out, dtype=torch.float32).to(device)
    # setup training profiles
    criterion = nn.MSELoss()
    criterion2 = nn.CosineEmbeddingLoss()
    if train_decoder:
        optimizer = torch.optim.Adam(list(student_encoder.parameters()) + list(student_decoder.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(student_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-8)
    # Train population model
    counter = 0
    teacher_encoder.eval()
    teacher_decoder.eval()
    student_decoder.eval()
    for epoch in range(epochs):
        temp_ind = torch.randperm(len(train_in1)).to(device)
        train_in1 = train_in1[temp_ind]
        train_in2 = train_in2[temp_ind]
        train_out = train_out[temp_ind]
        student_encoder.train()
        if train_decoder:
            student_decoder.train()
        # if epoch < 10:
        #     for param in student_decoder.parameters():
        #         param.requires_grad = False
        # else:
        #     for param in student_decoder.parameters():
        #         param.requires_grad = True
        epoch_loss = 0
        for i in range(0, len(train_in1), batch_size):
            temp_in1 = train_in1[i:i+batch_size]
            temp_in2 = train_in2[i:i+batch_size]
            temp_out = train_out[i:i+batch_size]
            if len(temp_in1) < batch_size:
                continue
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_mem = teacher_encoder(temp_in1)
                teacher_pred = teacher_decoder(teacher_mem)
            student_mem = student_encoder(temp_in2)
            if train_decoder:
                pred = student_decoder(student_mem)  # , mem_mask=mem_mask)
                loss_pred = criterion(pred, temp_out)
                temp_target = torch.ones(student_mem.size(0)).to(device)
                loss_KD = criterion2(student_mem[:, -1, :], teacher_mem[:, -1, :], target=temp_target)
                # loss_KD = criterion(student_mem[:,-1], teacher_mem[:,-1])
                loss_pp = criterion(pred, teacher_pred)
                # loss = (1 - align_loss_weight) *loss1 + align_loss_weight * loss2
                loss = loss_pred + 0.2 * loss_KD + 0.05 * loss_pp
                # print('losssss:', loss_pred.item(), loss_KD.item(), loss_pp.item(), loss.item())
                # loss = loss_KD + 0.2 * loss_pp
            else:
                temp_target = torch.ones(student_mem.size(0)).to(device)
                loss = criterion2(student_mem[:, -1, :], teacher_mem[:, -1, :], target=temp_target)
                # loss = criterion(student_mem[:, -1, :], teacher_mem[:, -1, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        student_encoder.eval()
        student_decoder.eval()
        val_loss = 0
        with torch.no_grad():  # validation
            for i in range(0, len(val_in1), 4096):
                student_mem = student_encoder(val_in2[i:i + 4096])
                outputs = student_decoder(student_mem)  # , mem_mask=mem_mask)
                # val_loss += criterion(outputs, val_out[i:i + 4096]).item()
                val_loss += mean_squared_error(outputs.cpu().numpy(), val_out[i:i + 4096].cpu().numpy())
        print(
            f'Epoch [{epoch + 1}/{epochs}], training loss:{epoch_loss:.6f}, val loss: {val_loss:.6f}, counter: {counter}')
        if epoch >= 10:
            if val_loss < best_val_loss:  # save model with lowest loss
                best_val_loss = val_loss
                counter = 0
                torch.save(student_encoder.state_dict(), 'best_encoder_model.pth')
                torch.save(student_decoder.state_dict(), 'best_decoder_model.pth')
                print("Best model saved")
            else:
                counter += 1
            if counter >= patience:  # early stop
                print(f'Early stopping triggerd after {epoch + 1} epochs. Validation loss: {best_val_loss:.6f}')
                student_encoder.load_state_dict(torch.load('best_encoder_model.pth'))
                student_decoder.load_state_dict(torch.load('best_decoder_model.pth'))
                print("Best model loaded after early stopping")
                break
            scheduler.step(val_loss)
    return student_encoder, student_decoder


def train_student_Encoder_Decoder(teacher_enc, teacher_dec, student_enc, student_dec, train_in, train_out, val_in, val_out, args, lr=1e-4, batch_size=512,
                   epochs=500, patients=20, warm_up=10, finetune=False, seq2seq=False):
    def warm_up_lambda(epoch):
        return float(epoch + 1) / float(warm_up)
    ##################### initialize optimization ################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_enc.to(device).to(torch.float32)
    teacher_dec.to(device).to(torch.float32)
    student_enc.to(device).to(torch.float32)
    student_dec.to(device).to(torch.float32)
    train_in = torch.tensor(train_in, dtype=torch.float32).to(device)
    train_out = torch.tensor(train_out, dtype=torch.float32).to(device)
    val_in = torch.tensor(val_in, dtype=torch.float32).to(device)
    val_out = torch.tensor(val_out, dtype=torch.float32).to(device)
    if finetune:
        optimizer = torch.optim.Adam(list(student_enc.parameters())+list(student_dec.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(student_enc.parameters(), lr=lr)
        for param in student_dec.parameters():
            param.requires_grad = False
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warm_up, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_lambda)
    if seq2seq:
        # out_std = torch.std(train_out, dim=0)
        median = train_out.median(dim=0).values
        mad = (train_out - median.unsqueeze(0)).abs().median(dim=0).values
        out_std = 1.4826 * mad
        criterion = StdweightedMSELoss(out_std)
    else:
        criterion = nn.MSELoss()
    criterion_kd = nn.L1Loss()
    counter = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        student_enc.train()
        student_dec.train()
        epoch_loss = 0.0
        for i in range(0, len(train_in), batch_size):
            temp_in = train_in[i:i+batch_size]
            temp_out = train_out[i:i+batch_size]
            if len(temp_in) < batch_size:
                temp_in = train_in[-batch_size:]
                temp_out = train_out[-batch_size:]
            with torch.no_grad():
                teacher_enc.eval()
                teacher_dec.eval()
                teacher_mem = teacher_enc(temp_in)
                teacher_pred = teacher_dec(teacher_mem)
                teacher_mem = teacher_mem.detach()
                teacher_pred = teacher_pred.detach()
            student_mem = student_enc(temp_in[:, :args.seq_len, :])
            student_pred = student_dec(student_mem)
            optimizer.zero_grad()
            loss_pred = criterion(temp_out, student_pred)
            loss_kd = criterion_kd(student_mem, teacher_mem)
            loss_soft = criterion(teacher_pred, student_pred)
            if finetune:
                loss = loss_pred + 0.8 * loss_kd + 0.5 * loss_soft
            else:
                loss = loss_kd
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(temp_in)
        epoch_loss /= len(train_in)

        val_loss = 0.0
        student_enc.eval()
        student_dec.eval()
        with torch.no_grad():
            for i in range(0, len(val_in), 4096):
                mem = student_enc(val_in[i:i+4096, :args.seq_len])
                pred = student_dec(mem)
                if seq2seq:
                    val_loss += criterion(pred, val_out[i:i+4096]).item()
                else:
                    val_loss += mean_squared_error(pred.cpu().numpy(), val_out[i:i+4096].cpu().numpy())
        print(
            f'Epoch [{epoch + 1}/{epochs}], training loss:{epoch_loss:.6f}, val loss: {val_loss:.6f}, counter: {counter}')
        # scheduler.step(val_loss)
        if epoch < warm_up:
            warmup_scheduler.step()
        else:
            scheduler.step()
            # scheduler.step(val_loss)
        if (val_loss < best_val_loss):  # save model with lowest loss
            best_val_loss = val_loss
            counter = 0
            torch.save(student_enc.state_dict(), 'best_enc_model.pth')
            torch.save(student_dec.state_dict(), 'best_dec_model.pth')
            print("Best model saved")
        else:
            counter += 1
        if counter >= patients:  # early stop
            print(f'Early stopping triggerd after {epoch + 1} epochs. Validation loss: {best_val_loss:.6f}')
            student_enc.load_state_dict(torch.load('best_enc_model.pth'))
            student_dec.load_state_dict(torch.load('best_dec_model.pth'))
            print("Best model loaded after early stopping")
            break
    return student_enc, student_dec