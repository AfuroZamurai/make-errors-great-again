import matplotlib.pyplot as plt

test_accuracy_extended = [0.3, 0.5, 0.8, 0.9, 0.93, 0.96]
train_accuracy_extended = [0.35, 0.63, 0.84, 0.91, 0.97, 0.99]

train_loss_extended = []
test_loss_extended = []

wer_ocr_extended = 0.25846483925159114
test_wer_extended = []

cer_ocr_extended = 0.052450439352967386
test_cer_extended = []

test_accuracy_unextended = []
train_accuracy_unextended = []

train_loss_unextended = []
test_loss_unextended = []

wer_ocr_unextended = 0.6746028800475059
test_wer_unextended = []

cer_ocr_unextended = 0.051398934330343475
test_cer_unextended = []


def plot(title, label_x, label_y, values1, values2, label_1, label_2):
    plt.plot(values1, label=label_1, color='blue')
    plt.plot(values2, label=label_2, color='orange')
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # plot results for expanded dataset
    plot('accuracy extended', 'epochs', 'accuracy', train_accuracy_extended,
         test_accuracy_extended, 'train accuracy', 'test accuracy')
    plot('loss extended', 'epochs', 'loss', train_loss_extended,
         test_loss_extended, 'train loss', 'test loss')
    plot('wer extended', 'epochs', 'wer', wer_ocr_extended,
         test_wer_extended, 'wer ocr', 'test wer')
    plot('cer extended', 'epochs', 'cer', cer_ocr_extended,
         test_cer_extended, 'cer ocr', 'test cer')

    # plot results for the unexpanded dataset
    plot('accuracy unextended', 'epochs', 'accuracy', train_accuracy_unextended,
         test_accuracy_unextended, 'train accuracy', 'test accuracy')
    plot('loss unextended', 'epochs', 'loss', train_loss_unextended,
         test_loss_unextended, 'train loss', 'test loss')
    plot('wer unextended', 'epochs', 'wer', wer_ocr_unextended,
         test_wer_unextended, 'wer ocr', 'test wer')
    plot('cer unextended', 'epochs', 'cer', cer_ocr_unextended,
         test_cer_unextended, 'cer ocr', 'test cer')

    # plot results together for unextended and extended dataset
    plot('test accuracy', 'epochs', 'accuracy', test_accuracy_unextended,
         test_accuracy_extended, 'test accuracy unxtended', 'test accuracy extended')
    plot('test loss', 'epochs', 'loss', test_loss_unextended,
         test_loss_extended, 'test loss unxtended', 'test loss extended')
    plot('wer', 'epochs', 'wer', test_wer_unextended,
         test_wer_extended, 'wer unextended', 'wer extended')
    plot('cer', 'epochs', 'cer', test_cer_unextended,
         test_cer_extended, 'cer unextended', 'cer extended')
