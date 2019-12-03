import code.torch.wer as wer
import codecs
import nltk


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def build_opts(cer=0):
    opts = AttributeDict()
    opts['vocab'] = None
    opts['equal_func'] = 'standard'
    opts['excp_file'] = None
    opts['v'] = False
    opts['V'] = 0
    opts['n'] = 0
    opts['cer'] = cer
    opts['ignore_blank'] = True
    opts['key_pressed'] = False
    opts['color'] = False
    return opts


def statistics(rec_dir, ref_dir, cer):
    rec_file = codecs.open(rec_dir, "rb", "utf-8", errors='ignore')
    ref_file = codecs.open(ref_dir, "rb", "utf-8", errors='ignore')
    opts = build_opts(cer=cer)
    stat = wer.calculate_statistics(rec_file, ref_file, opts)
    return stat


def char_error_rate(rec_sent, ref_sent):
    return nltk.edit_distance(rec_sent, ref_sent)/float(len(ref_sent))

def word_error_rate(rec_sent, ref_sent, ws):  # sent: list of ids, whitespace index

    rec_sent = [str(e) for e in rec_sent]
    ref_sent = [str(e) for e in ref_sent]
    rec_words = ''.join(rec_sent).split(str(ws))
    ref_words = ''.join(ref_sent).split(str(ws))
    return nltk.edit_distance(rec_words, ref_words)/float(len(ref_words))




def main():
    rec_dir = 'Z124117102_char_predict_0.txt'
    ref_dir = 'Z124117102_truth_0.txt'
    rec_file = codecs.open(rec_dir, "rb", "utf-8", errors='ignore')
    ref_file = codecs.open(ref_dir, "rb", "utf-8", errors='ignore')
    opts = build_opts()

    stat = wer.calculate_statistics(rec_file, ref_file, opts)
    print(stat)


if __name__ == '__main__':
    main()



