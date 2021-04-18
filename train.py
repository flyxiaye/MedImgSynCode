from CreateTrainer import trainer_factory


trainer = trainer_factory('cgan_grad_img', r'C:\Users\ChxxxXL\Documents\MedicalImage\test_trainer_data')
# trainer.prepare_data()
trainer.train(epochs=5, batch_size=1)