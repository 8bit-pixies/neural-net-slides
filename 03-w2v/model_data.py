from faker import Factory

fake = Factory.create()
 
# http://www.cupcakeipsum.com
word1 = """Dragee wafer cookie donut pudding cupcake lemon drops apple pie. Dessert topping ice cream pie powder carrot cake liquorice marzipan. Chocolate powder ice cream. Powder powder tart chocolate bar. Fruitcake chupa chups pastry dessert fruitcake ice cream apple pie apple pie. Cookie cotton candy lollipop. Chocolate candy canes lemon drops cotton candy pie caramels topping cotton candy. Cookie danish gingerbread danish sesame snaps pie. Sugar plum ice cream dessert lollipop pie pudding lemon drops carrot cake lemon drops. Cake jelly-o carrot cake bonbon cupcake donut. Chocolate bar lollipop dessert donut oat cake souffle. Pastry carrot cake gummi bears gummi bears lollipop pie gingerbread. Toffee chocolate souffle.
Chocolate souffle fruitcake. Cake sesame snaps wafer sugar plum sugar plum candy canes apple pie cake dessert. Macaroon jelly beans topping bonbon tiramisu brownie sesame snaps sweet roll fruitcake. Gummi bears pudding topping gummies cheesecake. Cupcake carrot cake sugar plum donut jujubes topping apple pie lemon drops. Cheesecake cheesecake liquorice bonbon. Tiramisu bonbon candy. Candy croissant lemon drops toffee powder marzipan. Apple pie cheesecake gummi bears. Apple pie jelly beans marzipan chocolate bar topping topping chocolate cake chocolate cake. Pie fruitcake bear claw muffin topping dessert donut chupa chups. Jelly beans marzipan brownie donut chupa chups sugar plum oat cake carrot cake pastry.
Jelly sesame snaps cheesecake chocolate cake caramels. Macaroon fruitcake lollipop halvah chocolate cake sugar plum cake. Sweet roll topping donut. Fruitcake pie sesame snaps sweet liquorice cake macaroon jelly. Lemon drops bonbon cupcake powder fruitcake gummies chupa chups gingerbread. Liquorice sugar plum ice cream. Sugar plum jelly-o gingerbread marzipan biscuit sesame snaps topping. Oat cake tiramisu caramels marzipan sugar plum souffle marshmallow. Cookie halvah halvah jelly-o cookie muffin jelly chocolate bar. Cheesecake carrot cake bear claw. Tiramisu pie jelly beans cotton candy. Powder dragee tart marzipan chupa chups brownie.""".split()
 
# http://www.catipsum.com
word2 = """Scream for no reason at 4 am stare at the wall, play with food and get confused by dust or poop in litter box, scratch the walls yet cats making all the muffins scratch the furniture catch mouse and gave it as a present small kitty warm kitty little balls of fur. Purr mrow. Chew on cable swat turds around the house. Bathe private parts with tongue then lick owner's face chase mice. Meow meow, i tell my human. Meowzer scratch the postman wake up lick paw wake up owner meow meow or attack feet throwup on your pillow climb leg, so Gate keepers of hell howl on top of tall thing. Behind the couch spill litter box, scratch at owner, destroy all furniture, especially couch curl into a furry donut but kitty scratches couch bad kitty cough furball. Mewl for food at 4am make muffins, or curl into a furry donut pooping rainbow while flying in a toasted bread costume in space, yet kitten is playing with dead mouse meow. Purr scratch the furniture stare at wall turn and meow stare at wall some more meow again continue staring . Lick sellotape pounce on unsuspecting person love and coo around boyfriend who purrs and makes the perfect moonlight eyes so i can purr and swat the glittery gleaming yarn to him (the yarn is from a $125 sweater). Chase red laser dot human give me attention meow yet ask for petting yet purr for scamper. Lick plastic bags sniff sniff cats go for world domination stare at ceiling. Purr cat is love, cat is life has closed eyes but still sees you. Eat from dog's food purrr purr littel cat, little cat purr purr. Cat is love, cat is life cough hairball on conveniently placed pants and sit in box. Jump five feet high and sideways when a shadow moves walk on car leaving trail of paw prints on hood and windshield for thinking longingly about tuna brine lies down but always hungry. Wack the mini furry mouse stick butt in face see owner, run in terror so inspect anything brought into the house. Intently stare at the same spot make muffins sleep in the bathroom sink. Roll on the floor purring your whiskers off behind the couch scream at teh bath stares at human while pushing stuff off a table mewl for food at 4am.
Rub face on everything demand to be let outside at once, and expect owner to wait for me as i think about it. Ears back wide eyed sleep on dog bed, force dog to sleep on floor. Scratch me there, elevator butt eat half my food and ask for more and sleep on keyboard find empty spot in cupboard and sleep all day mewl for food at 4am. Chase ball of string see owner, run in terror and attack the dog then pretend like nothing happened, so ignore the squirrels, you'll never catch them anyway. Where is my slave? I'm getting hungry destroy couch slap owner's face at 5am until human fills food dish scratch at the door then walk away yet spit up on light gray carpet instead of adjacent linoleum. Pelt around the house and up and down stairs chasing phantoms kitty scratches couch bad kitty yet chase after silly colored fish toys around the house fall asleep on the washing machine so lick the other cats sit in box.
""".split()

def create_data_sample(size=1000):
    data = []
    label = []
    for _ in range(size):
        data.append(fake.sentence(nb_words=20, ext_word_list=word1))
        label.append(0)
        data.append(fake.sentence(nb_words=20, ext_word_list=word2))
        label.append(1)
    return data, label