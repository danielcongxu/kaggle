import pandas as pd
import toxic
import severe_toxic
import obscene
import threat
import insult
import identity_hate
import misc


train_set = pd.read_csv("dataset/train.csv")
test_set = pd.read_csv("dataset/test.csv")


if __name__ == "__main__":
    # Generate vocabulary
    vocab = misc.sumForToxicType(train_set)

    comment_type = 'toxic'
    clf = toxic.train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True, skip_XGB=True, skip_ExtTree=True)
    result_toxic = toxic.predict(test_set, comment_type, vocab, clf, use_proba=True)

    comment_type = 'severe_toxic'
    clf = severe_toxic.train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True, skip_XGB=True, skip_ExtTree=True)
    result_severe_toxic = severe_toxic.predict(test_set, comment_type, vocab, clf, use_proba=True)

    comment_type = 'obscene'
    clf = obscene.train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True, skip_XGB=True, skip_ExtTree=True)
    result_obscene = obscene.predict(test_set, comment_type, vocab, clf, use_proba=True)

    comment_type = 'threat'
    clf = threat.train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True, skip_XGB=True, skip_ExtTree=True)
    result_threat = threat.predict(test_set, comment_type, vocab, clf, use_proba=True)

    comment_type = 'insult'
    clf = insult.train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True, skip_XGB=True, skip_ExtTree=True)
    result_insult = insult.predict(test_set, comment_type, vocab, clf, use_proba=True)

    comment_type = 'identity_hate'
    clf = identity_hate.train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True, skip_XGB=True, skip_ExtTree=True)
    result_identity_hate = identity_hate.predict(test_set, comment_type, vocab, clf, use_proba=True)

    submit = pd.DataFrame({'id': test_set.loc[:, 'id'],
                           'toxic': result_toxic,
                           'severe_toxic': result_severe_toxic,
                           'obscene': result_obscene,
                           'threat': result_threat,
                           'insult': result_insult,
                           'identity_hate': result_identity_hate})
    submit.to_csv('submit.csv', index=False)
