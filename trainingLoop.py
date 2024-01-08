import random
from binary_converter import float2bit
import torch
from matplotlib import pyplot as plt


def trainingLoop(PATH, print_model, do_training, device, batch_size, lookback, n_epochs, optimiser, model, loss_fn,
                 X_train,
                 X_test,
                 X_val, Y_train,
                 Y_test, Y_val):
    if print_model == True:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    else:
        pass
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    model = model.to(device)
    best_loss_train = 100000.
    best_loss_test = 100000.
    training_loss = []
    test_loss = []
    if do_training:
        for epoch in range(1, n_epochs + 1):
            optimiser.zero_grad()
            model.train(mode=True)
            loss_train = 0.
            loss_test = 0.
            indexes_train = random.sample(range(1, len(Y_train)), batch_size)
            indexes_test = random.sample(range(1, len(Y_test)), batch_size)
            output_train = model(X_train[indexes_train, :, :])
            output_test = model(X_test[indexes_test, :, :])

            for i in range(0, len(output_train)):
                for j in range(0, lookback):
                    # uSING RMSE LOSS (sqrt before mse)
                    # print(output_train[i].shape,torch.unsqueeze(Y_train[indexes_train,j, i], dim=1).shape)
                    target = torch.unsqueeze(Y_train[indexes_train, j, i], dim=1)
                    preds = output_train[i]

                    # print(preds[0],target[0],'\n','ddd')
                    loss_train += loss_fn(preds, target)  # calculate loss
                    loss_test += loss_fn(preds, target)

                    # print(output_train[i].shape,torch.unsqueeze(Y_train[indexes,i],dim=1).shape)
                    # time.sleep(0.5)

            loss_train = loss_train / len(output_train) / lookback
            loss_test = loss_test / len(output_test) / lookback

            loss_train.backward()
            optimiser.step()
            training_loss.append(loss_train.item())
            test_loss.append(loss_test.item())
            if loss_test < best_loss_test and loss_train < best_loss_train:
                torch.save(model.state_dict(), PATH + model.__class__.__name__ + '.pth')
                best_loss_train = loss_train
                best_loss_test = loss_test
            if epoch == 1 or epoch % 100 == 0:
                print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                      f" Test loss {loss_test.item():.4f}")
            if epoch % 500 == 0:
                # model.load_state_dict(torch.load(PATH + model.__class__.__name__ + '.pth'))
                model.train(mode=False)
                indexes_val = random.sample(range(1, len(Y_val)), batch_size)
                loss_val = 0.
                output_val = model(X_val[indexes_val])
                for i in range(0, len(output_train)):
                    for j in range(0, lookback):
                        loss_val += loss_fn(output_val[i], torch.unsqueeze(Y_val[indexes_val, j, i], dim=1))
                loss_val = loss_val / len(output_val) / lookback
                print(f"----\nEpoch {epoch}, Validation loss {loss_val.item():.4f}\n----")
        plt.plot(training_loss, label='training loss')
        plt.plot(test_loss, label='test loss')
        # plt.yscale('log')
        plt.grid()
        plt.legend(loc="best")
        plt.show()
    else:
        print("Training not performed for : ", model.__class__.__name__)
    print("Done!")
