import time
import torch

def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function, scheduler):

    model.train()
    total_loss = 0
    number = 0
    epoch_start = time.time()
    
    for _,batch  in enumerate(data_loader):

        # 获取图像和标签
        images, labels = batch['image'].to(device), batch['label'].to(device)  # [n,c*sw,w,h,d]

        # 每个batch梯度清0
        optimizer.zero_grad()

        # forward + loss
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # 反向传播 + 优化器更新参数
        loss.backward()
        optimizer.step()

        # 更新总loss和总batch数
        total_loss += loss.item()
        number += 1

        # 清空占用空间
        del images, labels
        torch.cuda.empty_cache()



    # 更新学习率
    scheduler.step()

    # 计算每轮平均loss
    epoch_mean_loss = total_loss / number

    # 打印此轮训练结果
    print(f"epoch {epoch + 1}, average loss: {epoch_mean_loss:.4f}, lr: {optimizer.state_dict()['param_groups'][0]['lr']}, step time: {(time.time() - epoch_start):.4f}")
    
    return epoch_mean_loss
