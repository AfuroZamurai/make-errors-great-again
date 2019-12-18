import matplotlib.pyplot as plt

test_accuracy_extended = [0.7070965451470282, 0.7804941924888245, 0.8112894980636469, 0.8024689677536508,
                          0.8245433257559468, 0.824237762052129, 0.8320524208794707,
                          0.8219011642215104, 0.826572441742985, 0.8339837644333767]
train_accuracy_extended = [0.6722365679052281, 0.968265541237086, 0.9756119234134826, 0.9785155923926042,
                           0.9802875369866807, 0.9814387061036284, 0.9823209723300838,
                           0.9830526061973055, 0.9835452347769523, 0.9840625609636177]

train_loss_extended = [1.179499476499192, 0.13732750316314932, 0.10398615467397185, 0.09049770881797191,
                       0.0821310056682904, 0.07698345101109368, 0.07302520662828414,
                       0.06952837361000526, 0.06727196750385801, 0.06501756276308374]
test_loss_extended = [2.525676090785791, 2.2180165628229047, 1.9890086175152486, 2.077954680583496,
                      1.9187175632560638, 2.0296052232112336, 1.912711366557162,
                      2.051451141881984, 2.0260692565102905, 1.9458246266873014]

wer_ocr_extended = 0.25846483925159114
test_wer_extended = [0.2105795766886435, 0.14182378026938555, 0.13022042565423447, 0.1312319579480142,
                     0.11217852153063344, 0.10801691661711607, 0.0956664285244345,
                     0.11968136892363687, 0.10453459321315767, 0.10702569251926243]

cer_ocr_extended = 0.052450439352967386
test_cer_extended = [0.04333597339745914, 0.02833611123740726, 0.023622144123360927, 0.02584639445307811,
                     0.021714793039798206, 0.022232950534573238, 0.019548139315742792,
                     0.023749341911131226, 0.022107437068461133, 0.021348036655782023]

test_accuracy_unextended = [0.7241226000445184, 0.7992901946428786, 0.824389157002743, 0.8319347040132062,
                            0.8420099334892964, 0.8465233170336939, 0.851904670425517,
                            0.8456579744372811, 0.8558572520500315, 0.857634703165227]
train_accuracy_unextended = [0.6467470979243485, 0.9709974858109999, 0.9793810707884342, 0.9822329788737478,
                             0.983809599437588, 0.9849088220605017, 0.9856135213105554,
                             0.9863379454595661, 0.9868362312697273, 0.9872065708784176]

train_loss_unextended = [1.2469871630933649, 0.13233949347582563, 0.0901996050276723, 0.07694533951292858,
                         0.06966969177551746, 0.06495380622285267, 0.06154998969026673,
                         0.058106927667504644, 0.05578188688411561, 0.05403075717115193]
test_loss_unextended = [2.310218718442668, 1.894968081398418, 1.7906743133719347, 1.7816491806946004,
                        1.6843413922254378, 1.7050553429650581, 1.6676856383433534,
                        1.753366956766031, 1.6451051678163422, 1.6676097597192416]

wer_ocr_unextended = 0.6746028800475059
test_wer_unextended = [0.5174250296912114, 0.39390959026128264, 0.35532957244655583, 0.33486119358669836,
                       0.35484709026128264, 0.31517220902612825, 0.3052998812351544,
                       0.3090669536817102, 0.31021748812351546, 0.30062351543942994]

cer_ocr_unextended = 0.051398934330343475
test_cer_unextended = [0.029883277240522713, 0.020657175985510378, 0.018254107675490152, 0.017779864447450795,
                       0.01809798481727154, 0.016374724344850627, 0.015691596620040105,
                       0.01630766182867183, 0.01611938347756658, 0.015604019527360822]


def plot(title, label_x, label_y, values1, values2, label_1, label_2, save_path='./results/', extension='.png'):
    plt.plot(values1, label=label_1, color='blue')
    plt.plot(values2, label=label_2, color='orange')
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend(loc='best')
    plt.savefig(save_path + title.replace(' ', '_') + extension)
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
