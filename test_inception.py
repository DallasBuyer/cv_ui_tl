import preprocess_imagess
import tensorflow as tf
import inference
import tensorlayer as tl
import model_utils
import time

X_train, y_train, X_test, y_test, size = preprocess_imagess.load_dataset("mnist_224_224_1")
X_holder = tf.placeholder(tf.float32, shape=[None, size[0], size[1], size[2]])
y_holder = tf.placeholder(tf.int64, shape=[None, ])
network = inference.load_model(inputs=X_holder, model_name="Inception_v1_224_224_1")
optimizer = model_utils.select_optimization(5, 0.001)

y = network.outputs
cost = tl.cost.cross_entropy(y, y_holder, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1),y_holder)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = optimizer.minimize(cost)

batch_size = 128
n_epoch = 10

x_train_batch, y_train_batch = tf.train.shuffle_batch([X_train, y_train],
                                                      batch_size=batch_size,
                                                      capacity=2000,
                                                      min_after_dequeue=1000,
                                                      num_threads=32)
x_test_batch, y_test_batch = tf.train.batch([X_test, y_test],
                                            batch_size=batch_size,
                                            capacity=50000,
                                            num_threads=32)
print "x_train_batch: ", x_train_batch, "\ny_train_batch: ", y_train_batch


sess = tf.InteractiveSession()

tl.layers.initialize_global_variables(sess)


network.print_params()
network.print_layers()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

n_step_epoch = int(50000 / batch_size)
n_step = n_epoch * n_step_epoch
step = 0
for epoch in range(n_epoch):
    start_time = time.time()
    train_loss, train_acc, n_batch = 0, 0, 0
    for s in range(n_step_epoch):
        val, l = sess.run([x_train_batch, y_train_batch])
        # print "@@"*10, s, "@@"*10, l
        # print type(val), val.dtype, type(l), l.dtype
        # l = np.array(l, dtype=np.float32)
        # print l.dtype
        err, ac, _ = sess.run([cost, acc, train_op], feed_dict={X_holder: val, y_holder: l})
        # err, ac, _ = sess.run([cost, acc, train_op])  # another way which don't use placeholder
        step += 1
        train_loss += err
        train_acc += ac
        n_batch += 1

    if epoch + 1 == 1 or epoch == n_epoch-1 or (epoch + 1) % 2 == 0:
        print ("Epoch %d: Step %d-%d of %d took %fs" %
               (epoch, step, step + n_epoch, n_step, time.time() - start_time))
        print ("train loss: %f" % (train_loss / n_batch))
        print ("train acc: %f" % (train_acc / n_batch))
        train_loss.append(train_loss/n_batch)
        train_acc.append(train_acc/n_batch)

        test_loss, test_acc, n_batch = 0, 0, 0
        for _ in range(int(10000 / batch_size)):
            val_test, l_test = sess.run([x_test_batch, y_test_batch])
            err, ac = sess.run([cost, acc], feed_dict={X_holder: val_test, y_holder: l_test})
            # err, ac = sess.run([cost_test, acc_test])   # another way which don't use placeholder
            test_loss += err
            test_acc += ac
            n_batch += 1
        print ("test loss: %f" % (test_loss / n_batch))
        print ("test acc: %f" % (test_acc / n_batch))
        test_loss.append(test_loss/n_batch)
        test_acc.append(test_acc/n_batch)


coord.request_stop()
coord.join(threads)
sess.close()
