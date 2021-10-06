 """ print(f"The count is: {t2_stop_cnt - t2_start}")
    x = model(TRAINING_INPUTS)
    some_testy = TRAINING_OUTPUT[0:10, :]
    # print(x.numpy()[0:10, :])
    y = x.numpy()
    one_hot_tensor = y
    label_train_prob = tf.argmax(one_hot_tensor, axis = 1)
    label2_train = tf.argmax(TRAINING_OUTPUT, axis=1)
    #print("Predicted labels")
    #print(label_train_prob)
    #print("Training labels")
    #print(label2_train)
    y = np.where(y < 0, 0, 1)
    print("10 point classifier output: ")
    print(y[0:10, :])
    print("10 point Test set output: ")
    print(some_testy)
    
    print("Confusion matrix for training set:")
    print(tf.math.confusion_matrix(label2_train, label_train_prob, num_classes=2).numpy())
    
    x_test = model(TESTING_INPUTS)
    some_testy = TRAINING_OUTPUT[0:10, :]
    # print(x.numpy()[0:10, :])
    y_test = x_test.numpy()
    one_hot_tensor_test = y_test
    label_test_prob = tf.argmax(one_hot_tensor_test, axis = 1)
    label2_test = tf.argmax(TESTING_OUTPUT, axis=1)
    print("The Confusion matrix for the test set: ")
    print(tf.math.confusion_matrix(label2_test, label_test_prob, num_classes=2).numpy())

    
    func_prediction_analysis(label_test_prob, label2_test)
 """