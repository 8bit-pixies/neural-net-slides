import numpy as np
import pandas as pd
 
 
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
 
 
from faker import Factory
 
 
# use example here:
# https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
from keras.preprocessing import sequence, text
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
 
 
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
 
 
# cnn settings
max_features = 5000
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2
 
 
model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=max_len))
# model.add(Dropout(0.2)) # this layer is a regularization layer
# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims, activation='relu', name='wordembedding')) # this is the word embedding if you wish to keep it, we can always extract it later
# model.add(Dropout(0.2)) # regularization layer
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2)) # as we have a classification problem the final layer is 2, if we have multi-class (say 3 classes then this would be `Dense(3)`.
model.add(Activation('softmax')) # if we use softmax, then we have softmax AKA multinomial regression
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=2,
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
# this will output your word embedding of size 250.
word_train = word_embedding.predict(x_train)
word_test = word_embedding.predict(x_test)
# you can verify using `word_train.shape` or `word_test.shape`