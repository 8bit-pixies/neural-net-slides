from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from faker import Factory
# use example here:
from keras.preprocessing import sequence, text
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
 
def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))
 
def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
     
    return new_sequences
fake = Factory.create()
 
# http://www.cupcakeipsum.com
word1 = """Dragee wafer cookie donut pudding cupcake lemon drops apple pie. Dessert topping ice cream pie powder carrot cake liquorice marzipan. Chocolate powder ice cream. Powder powder tart chocolate bar. Fruitcake chupa chups pastry dessert fruitcake ice cream apple pie apple pie. Cookie cotton candy lollipop. Chocolate candy canes lemon drops cotton candy pie caramels topping cotton candy. Cookie danish gingerbread danish sesame snaps pie. Sugar plum ice cream dessert lollipop pie pudding lemon drops carrot cake lemon drops. Cake jelly-o carrot cake bonbon cupcake donut. Chocolate bar lollipop dessert donut oat cake souffle. Pastry carrot cake gummi bears gummi bears lollipop pie gingerbread. Toffee chocolate souffle.
Chocolate souffle fruitcake. Cake sesame snaps wafer sugar plum sugar plum candy canes apple pie cake dessert. Macaroon jelly beans topping bonbon tiramisu brownie sesame snaps sweet roll fruitcake. Gummi bears pudding topping gummies cheesecake. Cupcake carrot cake sugar plum donut jujubes topping apple pie lemon drops. Cheesecake cheesecake liquorice bonbon. Tiramisu bonbon candy. Candy croissant lemon drops toffee powder marzipan. Apple pie cheesecake gummi bears. Apple pie jelly beans marzipan chocolate bar topping topping chocolate cake chocolate cake. Pie fruitcake bear claw muffin topping dessert donut chupa chups. Jelly beans marzipan brownie donut chupa chups sugar plum oat cake carrot cake pastry.
Jelly sesame snaps cheesecake chocolate cake caramels. Macaroon fruitcake lollipop halvah chocolate cake sugar plum cake. Sweet roll topping donut. Fruitcake pie sesame snaps sweet liquorice cake macaroon jelly. Lemon drops bonbon cupcake powder fruitcake gummies chupa chups gingerbread. Liquorice sugar plum ice cream. Sugar plum jelly-o gingerbread marzipan biscuit sesame snaps topping. Oat cake tiramisu caramels marzipan sugar plum souffle marshmallow. Cookie halvah halvah jelly-o cookie muffin jelly chocolate bar. Cheesecake carrot cake bear claw. Tiramisu pie jelly beans cotton candy. Powder dragee tart marzipan chupa chups brownie.""".split()
 
# http://www.catipsum.com
word2 = """Scream for no reason at 4 am stare at the wall, play with food and get confused by dust or poop in litter box, scratch the walls yet cats making all the muffins scratch the furniture catch mouse and gave it as a present small kitty warm kitty little balls of fur. Purr mrow. Chew on cable swat turds around the house. Bathe private parts with tongue then lick owner's face chase mice. Meow meow, i tell my human. Meowzer scratch the postman wake up lick paw wake up owner meow meow or attack feet throwup on your pillow climb leg, so Gate keepers of hell howl on top of tall thing. Behind the couch spill litter box, scratch at owner, destroy all furniture, especially couch curl into a furry donut but kitty scratches couch bad kitty cough furball. Mewl for food at 4am make muffins, or curl into a furry donut pooping rainbow while flying in a toasted bread costume in space, yet kitten is playing with dead mouse meow. Purr scratch the furniture stare at wall turn and meow stare at wall some more meow again continue staring . Lick sellotape pounce on unsuspecting person love and coo around boyfriend who purrs and makes the perfect moonlight eyes so i can purr and swat the glittery gleaming yarn to him (the yarn is from a $125 sweater). Chase red laser dot human give me attention meow yet ask for petting yet purr for scamper. Lick plastic bags sniff sniff cats go for world domination stare at ceiling. Purr cat is love, cat is life has closed eyes but still sees you. Eat from dog's food purrr purr littel cat, little cat purr purr. Cat is love, cat is life cough hairball on conveniently placed pants and sit in box. Jump five feet high and sideways when a shadow moves walk on car leaving trail of paw prints on hood and windshield for thinking longingly about tuna brine lies down but always hungry. Wack the mini furry mouse stick butt in face see owner, run in terror so inspect anything brought into the house. Intently stare at the same spot make muffins sleep in the bathroom sink. Roll on the floor purring your whiskers off behind the couch scream at teh bath stares at human while pushing stuff off a table mewl for food at 4am.
Rub face on everything demand to be let outside at once, and expect owner to wait for me as i think about it. Ears back wide eyed sleep on dog bed, force dog to sleep on floor. Scratch me there, elevator butt eat half my food and ask for more and sleep on keyboard find empty spot in cupboard and sleep all day mewl for food at 4am. Chase ball of string see owner, run in terror and attack the dog then pretend like nothing happened, so ignore the squirrels, you'll never catch them anyway. Where is my slave? I'm getting hungry destroy couch slap owner's face at 5am until human fills food dish scratch at the door then walk away yet spit up on light gray carpet instead of adjacent linoleum. Pelt around the house and up and down stairs chasing phantoms kitty scratches couch bad kitty yet chase after silly colored fish toys around the house fall asleep on the washing machine so lick the other cats sit in box.
""".split()
data = []
label = []
for _ in range(1000):
    data.append(fake.sentence(nb_words=20, ext_word_list=word1))
    label.append(0)
    data.append(fake.sentence(nb_words=20, ext_word_list=word2))
    label.append(1)
# determine the max length of sentence so we can pad it.
max_len = max([len(x.split()) for x in data])
# split in sklearn
x_train_text, x_test_text, y_train,  y_test = train_test_split(data, label, test_size=0.3, random_state=0)
# use keras tokenizer to convert sentence to sequence?
sent_to_seq = text.Tokenizer()
sent_to_seq.fit_on_texts(x_train_text)
x_train = sequence.pad_sequences(sent_to_seq.texts_to_sequences(x_train_text), maxlen=max_len)
x_test = sequence.pad_sequences(sent_to_seq.texts_to_sequences(x_test_text), maxlen=max_len)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
# config for model
ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 5
if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)
     
    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
     
    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1
     
    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')
model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(embedding_dims, activation='sigmoid', name='wordembedding')) # this line is not in the original fasttext but is here for transfer learning purposes
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
 
# show performance...this is rather crude but works in this instance
# normally we would want to set a threshold
yh_train = np.argmax(model.predict(x_train), axis=1)
yh_test = np.argmax(model.predict(x_test), axis=1)
# Accuracy should be very very high
print("Train accuracy: {}".format(accuracy_score(y_train, yh_train)))
print("Test accuracy: {}".format(accuracy_score(y_test, yh_test)))
# get the output as a vector:
word_embedding = Model(inputs=model.input,
                       outputs=model.get_layer(name='wordembedding').output)
# this will output your word embedding of size based on the variable `embedding_dims`.
word_train = word_embedding.predict(x_train)
word_test = word_embedding.predict(x_test)
# you can verify using `word_train.shape` or `word_test.shape`