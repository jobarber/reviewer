import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from peerreviewer import DEVICE


class Summarizer(torch.nn.Module):

    def __init__(self, pretrained_path="sshleifer/distilbart-cnn-12-6", max_length=256, stride=7):
        super(Summarizer, self).__init__()
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path).to(DEVICE)

    def forward(self, documents, labels=None):
        tokenized = self.tokenizer(documents, padding=True, return_tensors='pt', truncation=True,
                                   max_length=self.max_length, stride=self.stride).to(DEVICE)
        return self.model(labels=labels, **tokenized)

    def generate(self, document, **generate_kwargs):
        tokenized = self.tokenizer(document, padding=True, return_tensors='pt', truncation=True,
                                   max_length=self.max_length, stride=self.stride).to(DEVICE)
        tokenized.update(generate_kwargs)
        generated_sequences = self.model.generate(**tokenized)
        return ' '.join(self.tokenizer.decode(generated_tokens) for generated_tokens in generated_sequences)


if __name__ == '__main__':
    summarizer = Summarizer(max_length=64, stride=7)
    text = """Poetry in general seems to have sprung from two causes, each of them
            lying deep in our nature. First, the instinct of imitation is implanted
            in man from childhood, one difference between him and other animals
            being that he is the most imitative of living creatures, and through
            imitation learns his earliest lessons; and no less universal is the
            pleasure felt in things imitated. We have evidence of this in the facts
            of experience. Objects which in themselves we view with pain, we delight
            to contemplate when reproduced with minute fidelity: such as the forms
            of the most ignoble animals and of dead bodies. The cause of this again
            is, that to learn gives the liveliest pleasure, not only to philosophers
            but to men in general; whose capacity, however, of learning is more
            limited. Thus the reason why men enjoy seeing a likeness is, that in
            contemplating it they find themselves learning or inferring, and saying
            perhaps, 'Ah, that is he.' For if you happen not to have seen the
            original, the pleasure will be due not to the imitation as such, but to
            the execution, the colouring, or some such other cause.
            
            Imitation, then, is one instinct of our nature. Next, there is the
            instinct for 'harmony' and rhythm, metres being manifestly sections of
            rhythm. Persons, therefore, starting with this natural gift developed
            by degrees their special aptitudes, till their rude improvisations gave
            birth to Poetry.
            
            Poetry now diverged in two directions, according to the individual
            character of the writers. The graver spirits imitated noble actions, and
            the actions of good men. The more trivial sort imitated the actions of
            meaner persons, at first composing satires, as the former did hymns to
            the gods and the praises of famous men. A poem of the satirical kind
            cannot indeed be put down to any author earlier than Homer; though many
            such writers probably there were. But from Homer onward, instances
            can be cited,--his own Margites, for example, and other similar
            compositions. The appropriate metre was also here introduced; hence the
            measure is still called the iambic or lampooning measure, being that
            in which people lampooned one another. Thus the older poets were
            distinguished as writers of heroic or of lampooning verse.
            
            As, in the serious style, Homer is pre-eminent among poets, for he alone
            combined dramatic form with excellence of imitation, so he too first
            laid down the main lines of Comedy, by dramatising the ludicrous instead
            of writing personal satire. His Margites bears the same relation to
            Comedy that the Iliad and Odyssey do to Tragedy. But when Tragedy and
            Comedy came to light, the two classes of poets still followed their
            natural bent: the lampooners became writers of Comedy, and the Epic
            poets were succeeded by Tragedians, since the drama was a larger and
            higher form of art.
            
            Whether Tragedy has as yet perfected its proper types or not; and
            whether it is to be judged in itself, or in relation also to the
            audience,--this raises another question. Be that as it may, Tragedy--as
            also Comedy--was at first mere improvisation. The one originated with
            the authors of the Dithyramb, the other with those of the phallic songs,
            which are still in use in many of our cities. Tragedy advanced by slow
            degrees; each new element that showed itself was in turn developed.
            Having passed through many changes, it found its natural form, and there
            it stopped.
            """
    from time import time
    start = time()
    print(summarizer.generate(text))
    print(f'total time for generation: {time() - start} seconds')
