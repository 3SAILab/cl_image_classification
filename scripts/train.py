from trainers.trainer import trainer
import models

model = models.GoogLeNet
trainer(cuda=2,batch_size=32,model=model,epochs=30,save_path="GoogLeNet.pth")