from IPython.display import clear_output
!pip install big-sleep --upgrade
!nvidia-smi -L

from tqdm import trange
from IPython.display import Image, display, clear_output
from big_sleep import Imagine
import random as rnd

st = """When everyone you have ever loved is finally gone
When everything you have ever wanted is finally done with
When all of your nightmares are for a time obscured as by a shining brainless beacon
Or a blinding eclipse of the many terrible shapes of this world
When you are calm and joyful and finally entirely alone
Then in a great new darkness you will finally execute your special plan
One needs to have a plan someone said who was turned away into the shadows
And who I had believed was sleeping or dead
Imagine he said all the flesh that is eaten
The teeth tearing into it the tongue tasting its savour and the hunger for that taste
Now take away that flesh he said
Take away the teeth and the tongue the taste and the hunger
Take away everything as it is
That was my plan my own special plan for this world
I listened to these words and yet I did not wonder
If this creature whom I had thought sleeping or dead would ever approach his vision
Even in his deepest dreams or his most lasting death
Because I had heard of such plans such visions and I knew they did not see far enough
But what was demanded in a way of a plan
Needed to go beyond tongue and teeth and hunger and flesh
Beyond the bones and the very dust of bones and the wind that would come to blow the dust away
And so I began to envision a darkness that was long before the dark of night
And a strangely shining light that owed nothing to the light of day
That day may seem like other days
Once more we feel the tinylegged trepidations once more we are mangled by a great grinding fear
But that day will have no others after no more worlds like this will follow
Because I have a plan a very special plan
No more worlds like this no more days like that
There are but four ways to die a sardonic spirit might have said to me
There is dying that occurs relatively suddenly
There is dying that occurs relatively gradually
There is dying that occurs relatively painlessly
There is the death that is full of pain
Thus by various means they are combined
The sudden and the gradual the painless and the painful
To yield but four ways to die and there are no others
Even after the voice stopped speaking I listened for it to speak again
After hours and day and years had passed I listened for some further words
Yet all I heard were the faintest echoes reminding me there are no others
Was it then that I began to conceive for this world a special plan
There are no means for escaping this world it penetrates even into your sleep
And is its substance you are caught in your own dreaming
Where there is no space and are held forever where there is no time
You can do nothing you are not told to do
There is no hope for escape from this dream that was never yours
The very words you speak are only its very words
And you talk like a traitor under its incessant torture
There are many who have designs upon this world and dream of wild and vast reformations
I have heard them talking in their sleep of elegant mutations and cunning annihilations
I have heard them whispering in the corners of crooked houses
And in the alleys and narrow back streets of this crooked creaking universe
Which they with their new designs would make straight and sound
But each of these new and illconceived designs is deranged in its heart
For they see this world as if it were alone and original
And not as only one of countless others whose nightmares all proceed
Like a hideous garden grown from a single seed
I have heard these dreamers talking in their sleep
And I stand waiting for them as at the top of a darkened flight of stairs
They know nothing of me and none of the secrets of my special plan
While I know every crooked creaking step of theirs
It was the voice of someone who was waiting in the shadows
Who was looking at the moon and waiting for me to turn the corner and enter a narrow street
And stand with him in the dull glaze of moonlight
Then he said to me he whispered that my plan was misconceived
That my special plan for this world was a terrible mistake
Because he said there is nothing to do and there is no where to go
There is nothing to be and there is no one to know
Your plan is a mistake he repeated
This world is a mistake I replied
The children always followed him when they saw him hopping by
A funny walk a funny man a funny funny funny man
He made them laugh sometimes
He made them laugh oh yes he did he did he did he did he did
Oh how he made them roll
One day he took them to a place he knew a special place
And told them things about this world this funny funny funny world
which made them laugh sometimes
Then the funny little man who made them laugh sometimes he did
Revealed to them his special plan his very special funny plan
Knowing they would understand and maybe laugh sometimes
He made them laugh oh yes he did he did he did he did he did
Their eyes grew wide beneath their lids
And how he made them roll
I first learned the facts from a lunatic
In a dark and quiet room that smelled of stale time and space
There are no people nothing at all like that
The human phenomenon is but the sum of densely coiled layers of illusion
Each of which winds itself upon the supreme insanity
That there are persons of any kind when all there can be is mindless mirrors
Laughing and screaming as they parade about in an endless dream
But when I asked the lunatic what it was that saw itself within these mirrors
As they marched endlessly in stale time and space
He only rocked and smiled then he laughed and screamed
And in his black and empty eyes I saw for a moment as in a mirror
A formless shade of divinity in flight from its stale infinity
Of time and space and the worst of all of this worlds dreams
My special plan for the laughter and the screams
We went to see some little show that was staged in an old shed past the edge of town
And in its beginnings all seemed well
The miniature curtain stage glowed in the darkness
While those dolls bounced along on their strings before our eyes
And in its beginnings all seemed well but then there came a subtle turning point
Which some had noticed and I was one who quietly left the show no I did not
Because I could see where things were going
As the antics of those dolls grew strange
And the fragile strings grew taut with the tiny pullings of tiny limbs
The others around me became appalled and turned away and abandoned the show
That was staged in an old shed past the edge of town
But I wanted to witness what could never be I wanted to see what could not be seen
But the moment of consummate disaster when puppets turn to face the puppet master
It was twilight and I stood in the greyish haze of a vast empty building
When the silence was enriched by a reverberant voice
All the things of this world it said
Are of but one essence for which there are no words
This is the greater part which has no beginning or end
And the one essence of this world for which there can be no words
Is but all the things of this world
This is the lesser part which had a beginning and shall have an end
And for which words were conceived solely to speak of
The tiny broken beings of this world it said
The beginnings and endings of this world it said
For which words were conceived solely to speak of
Now remove these words and what remains it asks me
As I stood in the twilight of that vast empty building
But I did not answer
The question echoed over and over but I remained silent until the echoes died
And as twilight passed into evening I felt my special plan
For which there are no words
moving towards a greater darkness
There are some who have no voices or none that will ever speak
Because of the things they know about this world
And the things they feel about this world
Because the thoughts that fill a brain that is a damaged brain
Because the pain that fills a body that is a damaged body
Exist in other worlds countless other worlds
Each of which stands alone in an infinite empty blackness
For which no words have been conceived and where no voices are able to speak
When a brain is filled only with damaged thoughts when a damaged body is filled only with pain
And stands alone in a world surrounded by infinite empty blackness
And exists in a world for which there is no special plan"""

#-----Experimental stuff:
#bilinear = False #not working
#experimental_resample = False #not working
# only if no seed value and random = true????
Center_bias = False #@param {type:"boolean"}
Torch_deterministic = False #@param {type:"boolean"}
Class_temperature = 2. #@param {type:"number"} #def "2."; Maybe weight of classes found in Description?
Ema_decay =  0.5#@param {type:"number"} #def "0.5"; Exponential Moving Average, Fuck if I know... Weird GANNN stuff
#image_size = 512 # can't be higher, only lower: 128, 256, 512
#gradient_accumulate_every = 1 #only 1 makes sense. above exponentially increases
#time needed to finish iteration. Also seems to have negative effect on accuracy
# vs description

for x in st.splitlines():
  Seed = rnd.randint(0, 1000000000)
  model = Imagine(
    #bilinear = bilinear,
    #experimental_resample = experimental_resample,
    #gradient_accumulate_every = gradient_accumulate_every,
    #image_size = image_size,
    torch_deterministic = Torch_deterministic,
    ema_decay = Ema_decay,
    class_temperature = Class_temperature,
    center_bias = Center_bias,
    text = x,
    save_every = 1,
    save_progress = True,
    lr = 0.07,
    text_min="",
    iterations = 26,
    epochs = 1,
    max_classes = 15,
    num_cutouts = 48,
    save_best = False,
    seed = rnd.randint(0, 1000000000),
    append_seed = False
  )

  for epoch in trange(1, desc = 'epochs'):
    for i in trange(26, desc = 'iteration'):
      path = f'{x}.{i}'
      model.train_step(epoch, i)

      if i == 0 or i % model.save_every != 0:  #basically: if i not multiple of save_every, skip next steps
        continue
      print("\nCurrent seed is: %i" % Seed)
  !mkdir "{x}"
  !mv ./*.png "{x}"
  !tar -zcvf "{x}.tar.gz" "{x}"
  !rm -rf "{x}"