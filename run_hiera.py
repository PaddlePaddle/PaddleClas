import sys

sys.path.append(
    "/media/user/40da1a25-a924-43a2-b274-e33d39ea8680/cv/lzy/paddle_hiera-main"
)
import os
import warnings

import paddle
from paddle_utils import *

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
from datetime import datetime

import hiera
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_model():
    """加载Hiera模型"""
    print("Loading Hiera model...")
    model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model, device


def get_transforms():
    """获取图像变换"""
    input_size = 224
    transform_list = [
        paddle.vision.transforms.Resize(size=int(256 / 224 * input_size)),
        paddle.vision.transforms.CenterCrop(input_size),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
    return paddle.vision.transforms.Compose(transform_list)


def get_imagenet_labels():
    """获取完整的ImageNet-1K标签"""
    imagenet_labels = {}
    imagenet_labels[0] = "tench, Tinca tinca"
    imagenet_labels[1] = "goldfish, Carassius auratus"
    imagenet_labels[
        2
    ] = "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias"
    imagenet_labels[3] = "tiger shark, Galeocerdo cuvieri"
    imagenet_labels[4] = "hammerhead, hammerhead shark"
    imagenet_labels[5] = "electric ray, crampfish, numbfish, torpedo"
    imagenet_labels[6] = "stingray"
    imagenet_labels[7] = "cock"
    imagenet_labels[8] = "hen"
    imagenet_labels[9] = "ostrich, Struthio camelus"
    imagenet_labels[10] = "brambling, Fringilla montifringilla"
    imagenet_labels[11] = "goldfinch, Carduelis carduelis"
    imagenet_labels[12] = "house finch, linnet, Carpodacus mexicanus"
    imagenet_labels[13] = "junco, snowbird"
    imagenet_labels[14] = "indigo bunting, indigo finch, indigo bird, Passerina cyanea"
    imagenet_labels[15] = "robin, American robin, Turdus migratorius"
    imagenet_labels[16] = "bulbul"
    imagenet_labels[17] = "jay"
    imagenet_labels[18] = "magpie"
    imagenet_labels[19] = "chickadee"
    imagenet_labels[20] = "water ouzel, dipper"
    imagenet_labels[21] = "kite"
    imagenet_labels[22] = "bald eagle, American eagle, Haliaeetus leucocephalus"
    imagenet_labels[23] = "vulture"
    imagenet_labels[24] = "great grey owl, great gray owl, Strix nebulosa"
    imagenet_labels[25] = "European fire salamander, Salamandra salamandra"
    imagenet_labels[26] = "common newt, Triturus vulgaris"
    imagenet_labels[27] = "eft"
    imagenet_labels[28] = "spotted salamander, Ambystoma maculatum"
    imagenet_labels[29] = "axolotl, mud puppy, Ambystoma mexicanum"
    imagenet_labels[30] = "bullfrog, Rana catesbeiana"
    imagenet_labels[31] = "tree frog, tree-frog"
    imagenet_labels[
        32
    ] = "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui"
    imagenet_labels[33] = "loggerhead, loggerhead turtle, Caretta caretta"
    imagenet_labels[
        34
    ] = "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea"
    imagenet_labels[35] = "mud turtle"
    imagenet_labels[36] = "terrapin"
    imagenet_labels[37] = "box turtle, box tortoise"
    imagenet_labels[38] = "banded gecko"
    imagenet_labels[39] = "common iguana, iguana, Iguana iguana"
    imagenet_labels[40] = "American chameleon, anole, Anolis carolinensis"
    imagenet_labels[41] = "whiptail, whiptail lizard"
    imagenet_labels[42] = "agama"
    imagenet_labels[43] = "frilled lizard, Chlamydosaurus kingi"
    imagenet_labels[44] = "alligator lizard"
    imagenet_labels[45] = "Gila monster, Heloderma suspectum"
    imagenet_labels[46] = "green lizard, Lacerta viridis"
    imagenet_labels[47] = "African chameleon, Chamaeleo chamaeleon"
    imagenet_labels[
        48
    ] = "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis"
    imagenet_labels[49] = "African crocodile, Nile crocodile, Crocodylus niloticus"
    imagenet_labels[50] = "American alligator, Alligator mississipiensis"
    imagenet_labels[51] = "triceratops"
    imagenet_labels[52] = "thunder snake, worm snake, Carphophis amoenus"
    imagenet_labels[53] = "ringneck snake, ring-necked snake, ring snake"
    imagenet_labels[54] = "hognose snake, puff adder, sand viper"
    imagenet_labels[55] = "green snake, grass snake"
    imagenet_labels[56] = "king snake, kingsnake"
    imagenet_labels[57] = "garter snake, grass snake"
    imagenet_labels[58] = "water snake"
    imagenet_labels[59] = "vine snake"
    imagenet_labels[60] = "night snake, Hypsiglena torquata"
    imagenet_labels[61] = "boa constrictor, Constrictor constrictor"
    imagenet_labels[62] = "rock python, rock snake, Python sebae"
    imagenet_labels[63] = "Indian cobra, Naja naja"
    imagenet_labels[64] = "green mamba"
    imagenet_labels[65] = "sea snake"
    imagenet_labels[
        66
    ] = "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus"
    imagenet_labels[67] = "diamondback, diamondback rattlesnake, Crotalus adamanteus"
    imagenet_labels[68] = "sidewinder, horned rattlesnake, Crotalus cerastes"
    imagenet_labels[69] = "trilobite"
    imagenet_labels[70] = "harvestman, daddy longlegs, Phalangium opilio"
    imagenet_labels[71] = "scorpion"
    imagenet_labels[72] = "black and gold garden spider, Argiope aurantia"
    imagenet_labels[73] = "barn spider, Araneus cavaticus"
    imagenet_labels[74] = "garden spider, Aranea diademata"
    imagenet_labels[75] = "black widow, Latrodectus mactans"
    imagenet_labels[76] = "tarantula"
    imagenet_labels[77] = "wolf spider, hunting spider"
    imagenet_labels[78] = "tick"
    imagenet_labels[79] = "centipede"
    imagenet_labels[80] = "black grouse"
    imagenet_labels[81] = "ptarmigan"
    imagenet_labels[82] = "ruffed grouse, partridge, Bonasa umbellus"
    imagenet_labels[83] = "prairie chicken, prairie grouse, prairie fowl"
    imagenet_labels[84] = "peacock"
    imagenet_labels[85] = "quail"
    imagenet_labels[86] = "partridge"
    imagenet_labels[87] = "African grey, African gray, Psittacus erithacus"
    imagenet_labels[88] = "macaw"
    imagenet_labels[89] = "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita"
    imagenet_labels[90] = "lorikeet"
    imagenet_labels[91] = "coucal"
    imagenet_labels[92] = "bee eater"
    imagenet_labels[93] = "hornbill"
    imagenet_labels[94] = "hummingbird"
    imagenet_labels[95] = "jacamar"
    imagenet_labels[96] = "toucan"
    imagenet_labels[97] = "drake"
    imagenet_labels[98] = "red-breasted merganser, Mergus serrator"
    imagenet_labels[99] = "goose"
    imagenet_labels[100] = "black swan, Cygnus atratus"
    imagenet_labels[101] = "tusker"
    imagenet_labels[102] = "echidna, spiny anteater, anteater"
    imagenet_labels[
        103
    ] = "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus"
    imagenet_labels[104] = "wallaby, brush kangaroo"
    imagenet_labels[
        105
    ] = "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus"
    imagenet_labels[106] = "wombat"
    imagenet_labels[107] = "jellyfish"
    imagenet_labels[108] = "sea anemone, anemone"
    imagenet_labels[109] = "brain coral"
    imagenet_labels[110] = "flatworm, platyhelminth"
    imagenet_labels[111] = "nematode, nematode worm, roundworm"
    imagenet_labels[112] = "conch"
    imagenet_labels[113] = "snail"
    imagenet_labels[114] = "slug"
    imagenet_labels[115] = "sea slug, nudibranch"
    imagenet_labels[116] = "chiton, coat-of-mail shell, sea cradle, polyplacophore"
    imagenet_labels[117] = "chambered nautilus, pearly nautilus, nautilus"
    imagenet_labels[118] = "Dungeness crab, Cancer magister"
    imagenet_labels[119] = "rock crab, Cancer irroratus"
    imagenet_labels[120] = "fiddler crab"
    imagenet_labels[
        121
    ] = "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica"
    imagenet_labels[
        122
    ] = "American lobster, Northern lobster, Maine lobster, Homarus americanus"
    imagenet_labels[
        123
    ] = "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish"
    imagenet_labels[124] = "crayfish, crawfish, crawdad, crawdaddy"
    imagenet_labels[125] = "hermit crab"
    imagenet_labels[126] = "isopod"
    imagenet_labels[127] = "white stork, Ciconia ciconia"
    imagenet_labels[128] = "black stork, Ciconia nigra"
    imagenet_labels[129] = "spoonbill"
    imagenet_labels[130] = "flamingo"
    imagenet_labels[131] = "little blue heron, Egretta caerulea"
    imagenet_labels[132] = "great blue heron, Ardea herodias"
    imagenet_labels[133] = "great egret, Egretta alba"
    imagenet_labels[134] = "snowy egret, Egretta thula"
    imagenet_labels[135] = "little egret, Egretta garzetta"
    imagenet_labels[136] = "bittern"
    imagenet_labels[137] = "crane"
    imagenet_labels[138] = "limpkin, Aramus pictus"
    imagenet_labels[139] = "European gallinule, Porphyrio porphyrio"
    imagenet_labels[140] = "American coot, marsh hen, mud hen, Fulica americana"
    imagenet_labels[141] = "bustard"
    imagenet_labels[142] = "ruddy turnstone, Arenaria interpres"
    imagenet_labels[143] = "red-backed sandpiper, dunlin, Erolia alpina"
    imagenet_labels[144] = "redshank, Tringa totanus"
    imagenet_labels[145] = "dowitcher"
    imagenet_labels[146] = "oystercatcher, oyster catcher"
    imagenet_labels[147] = "pelican"
    imagenet_labels[148] = "king penguin, Aptenodytes patagonica"
    imagenet_labels[149] = "albatross, mollymawk"
    imagenet_labels[
        150
    ] = "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus"
    imagenet_labels[151] = "killer whale, killer, orca, grampus, sea wolf, Orcinus orca"
    imagenet_labels[152] = "dugong, Dugong dugon"
    imagenet_labels[153] = "sea lion"
    imagenet_labels[154] = "Chihuahua"
    imagenet_labels[155] = "Japanese spaniel"
    imagenet_labels[156] = "Maltese dog, Maltese terrier, Maltese"
    imagenet_labels[157] = "Pekinese, Pekingese, Peke"
    imagenet_labels[158] = "Shih-Tzu"
    imagenet_labels[159] = "Blenheim spaniel"
    imagenet_labels[160] = "papillon"
    imagenet_labels[161] = "toy terrier"
    imagenet_labels[162] = "Rhodesian ridgeback"
    imagenet_labels[163] = "Afghan hound, Afghan"
    imagenet_labels[164] = "basset, basset hound"
    imagenet_labels[165] = "beagle"
    imagenet_labels[166] = "bloodhound, sleuthhound"
    imagenet_labels[167] = "bluetick"
    imagenet_labels[168] = "black-and-tan coonhound"
    imagenet_labels[169] = "Walker hound, Walker foxhound"
    imagenet_labels[170] = "English foxhound"
    imagenet_labels[171] = "redbone"
    imagenet_labels[172] = "borzoi, Russian wolfhound"
    imagenet_labels[173] = "Irish wolfhound"
    imagenet_labels[174] = "Italian greyhound"
    imagenet_labels[175] = "whippet"
    imagenet_labels[176] = "Ibizan hound, Ibizan Podenco"
    imagenet_labels[177] = "Norwegian elkhound, elkhound"
    imagenet_labels[178] = "otterhound, otter hound"
    imagenet_labels[179] = "Saluki, gazelle hound"
    imagenet_labels[180] = "Scottish deerhound, deerhound"
    imagenet_labels[181] = "Weimaraner"
    imagenet_labels[182] = "Staffordshire bullterrier, Staffordshire bull terrier"
    imagenet_labels[
        183
    ] = "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier"
    imagenet_labels[184] = "Bedlington terrier"
    imagenet_labels[185] = "Border terrier"
    imagenet_labels[186] = "Kerry blue terrier"
    imagenet_labels[187] = "Irish terrier"
    imagenet_labels[188] = "Norfolk terrier"
    imagenet_labels[189] = "Norwich terrier"
    imagenet_labels[190] = "Yorkshire terrier"
    imagenet_labels[191] = "wire-haired fox terrier"
    imagenet_labels[192] = "Lakeland terrier"
    imagenet_labels[193] = "Sealyham terrier, Sealyham"
    imagenet_labels[194] = "Airedale, Airedale terrier"
    imagenet_labels[195] = "cairn, cairn terrier"
    imagenet_labels[196] = "Australian terrier"
    imagenet_labels[197] = "Dandie Dinmont, Dandie Dinmont terrier"
    imagenet_labels[198] = "Boston bull, Boston terrier"
    imagenet_labels[199] = "miniature schnauzer"
    imagenet_labels[200] = "giant schnauzer"
    imagenet_labels[201] = "standard schnauzer"
    imagenet_labels[202] = "Scotch terrier, Scottish terrier, Scottie"
    imagenet_labels[203] = "Tibetan terrier, chrysanthemum dog"
    imagenet_labels[204] = "silky terrier, Sydney silky"
    imagenet_labels[205] = "soft-coated wheaten terrier"
    imagenet_labels[206] = "West Highland white terrier"
    imagenet_labels[207] = "Chesapeake Bay retriever (切萨皮克湾寻回犬)"
    imagenet_labels[208] = "Curly-coated retriever"
    imagenet_labels[209] = "flat-coated retriever"
    imagenet_labels[210] = "golden retriever"
    imagenet_labels[211] = "Labrador retriever"
    imagenet_labels[212] = "German short-haired pointer"
    imagenet_labels[213] = "vizsla, Hungarian pointer"
    imagenet_labels[214] = "English setter"
    imagenet_labels[215] = "Irish setter, red setter"
    imagenet_labels[216] = "Gordon setter"
    imagenet_labels[217] = "Brittany spaniel"
    imagenet_labels[218] = "clumber, clumber spaniel"
    imagenet_labels[219] = "English springer, English springer spaniel"
    imagenet_labels[220] = "Welsh springer spaniel"
    imagenet_labels[221] = "cocker spaniel, English cocker spaniel, cocker"
    imagenet_labels[222] = "Sussex spaniel"
    imagenet_labels[223] = "Irish water spaniel"
    imagenet_labels[224] = "kuvasz"
    imagenet_labels[225] = "schipperke"
    imagenet_labels[226] = "groenendael"
    imagenet_labels[227] = "malinois"
    imagenet_labels[228] = "briard"
    imagenet_labels[229] = "kelpie"
    imagenet_labels[230] = "komondor"
    imagenet_labels[231] = "Old English sheepdog, bobtail"
    imagenet_labels[232] = "Shetland sheepdog, Shetland sheep dog, Shetland"
    imagenet_labels[233] = "collie"
    imagenet_labels[234] = "Border collie"
    imagenet_labels[235] = "Bouvier des Flandres, Bouviers des Flandres"
    imagenet_labels[236] = "Rottweiler"
    imagenet_labels[
        237
    ] = "German shepherd, German shepherd dog, German police dog, alsatian"
    imagenet_labels[238] = "Doberman, Doberman pinscher"
    imagenet_labels[239] = "miniature pinscher"
    imagenet_labels[240] = "Greater Swiss Mountain dog"
    imagenet_labels[241] = "Bernese mountain dog"
    imagenet_labels[242] = "Appenzeller"
    imagenet_labels[243] = "golden retriever (金毛犬)"
    imagenet_labels[244] = "Labrador retriever (拉布拉多犬)"
    imagenet_labels[245] = "EntleBucher"
    imagenet_labels[246] = "boxer"
    imagenet_labels[247] = "bull mastiff"
    imagenet_labels[248] = "Tibetan mastiff"
    imagenet_labels[249] = "French bulldog"
    imagenet_labels[250] = "Great Dane"
    imagenet_labels[251] = "Saint Bernard, St Bernard"
    imagenet_labels[252] = "Eskimo dog, husky"
    imagenet_labels[253] = "malamute, malemute, Alaskan malamute"
    imagenet_labels[254] = "Siberian husky"
    imagenet_labels[255] = "dalmatian, coach dog, carriage dog"
    imagenet_labels[256] = "affenpinscher, monkey pinscher, monkey dog"
    imagenet_labels[257] = "basenji"
    imagenet_labels[258] = "pug, pug-dog"
    imagenet_labels[259] = "Leonberg"
    imagenet_labels[260] = "Newfoundland, Newfoundland dog"
    imagenet_labels[261] = "Great Pyrenees"
    imagenet_labels[262] = "Samoyed, Samoyede"
    imagenet_labels[263] = "Pomeranian"
    imagenet_labels[264] = "chow, chow chow"
    imagenet_labels[265] = "keeshond"
    imagenet_labels[266] = "Brabancon griffon"
    imagenet_labels[267] = "Pembroke, Pembroke Welsh corgi"
    imagenet_labels[268] = "Cardigan, Cardigan Welsh corgi"
    imagenet_labels[269] = "toy poodle"
    imagenet_labels[270] = "miniature poodle"
    imagenet_labels[271] = "standard poodle"
    imagenet_labels[272] = "Mexican hairless"
    imagenet_labels[273] = "timber wolf, grey wolf, gray wolf, Canis lupus"
    imagenet_labels[274] = "white wolf, Arctic wolf, Canis lupus tundrarum"
    imagenet_labels[275] = "red wolf, maned wolf, Canis rufus, Canis niger"
    imagenet_labels[276] = "coyote, prairie wolf, brush wolf, Canis latrans"
    imagenet_labels[277] = "dingo, warrigal, warragal, Canis dingo"
    imagenet_labels[278] = "dhole, Cuon alpinus"
    imagenet_labels[
        279
    ] = "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus"
    imagenet_labels[280] = "hyena, hyaena"
    imagenet_labels[281] = "red fox, Vulpes vulpes"
    imagenet_labels[282] = "kit fox, Vulpes macrotis"
    imagenet_labels[283] = "Arctic fox, white fox, Alopex lagopus"
    imagenet_labels[284] = "grey fox, gray fox, Urocyon cinereoargenteus"
    imagenet_labels[285] = "tabby, tabby cat"
    imagenet_labels[286] = "tiger cat"
    imagenet_labels[287] = "Persian cat"
    imagenet_labels[288] = "Siamese cat, Siamese"
    imagenet_labels[289] = "Egyptian cat"
    imagenet_labels[
        290
    ] = "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor"
    imagenet_labels[291] = "lynx, catamount"
    imagenet_labels[292] = "leopard, Panthera pardus"
    imagenet_labels[293] = "snow leopard, ounce, Panthera uncia"
    imagenet_labels[294] = "jaguar, panther, Panthera onca, Felis onca"
    imagenet_labels[295] = "lion, king of beasts, Panthera leo"
    imagenet_labels[296] = "tiger, Panthera tigris"
    imagenet_labels[297] = "cheetah, chetah, Acinonyx jubatus"
    imagenet_labels[298] = "brown bear, bruin, Ursus arctos"
    imagenet_labels[
        299
    ] = "American black bear, black bear, Ursus americanus, Euarctos americanus"
    imagenet_labels[300] = "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus"
    imagenet_labels[301] = "sloth bear, Melursus ursinus, Ursus ursinus"
    imagenet_labels[302] = "mongoose"
    imagenet_labels[303] = "meerkat, mierkat"
    imagenet_labels[304] = "tiger beetle"
    imagenet_labels[305] = "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle"
    imagenet_labels[306] = "ground beetle, carabid beetle"
    imagenet_labels[307] = "long-horned beetle, longicorn, longicorn beetle"
    imagenet_labels[308] = "leaf beetle, chrysomelid"
    imagenet_labels[309] = "dung beetle"
    imagenet_labels[310] = "rhinoceros beetle"
    imagenet_labels[311] = "weevil"
    imagenet_labels[312] = "fly"
    imagenet_labels[313] = "bee"
    imagenet_labels[314] = "ant, emmet, pismire"
    imagenet_labels[315] = "grasshopper, hopper"
    imagenet_labels[316] = "cricket"
    imagenet_labels[317] = "walking stick, walkingstick, stick insect"
    imagenet_labels[318] = "cockroach, roach"
    imagenet_labels[319] = "mantis, mantid"
    imagenet_labels[320] = "cicada, cicala"
    imagenet_labels[321] = "leafhopper"
    imagenet_labels[322] = "lacewing, lacewing fly"
    imagenet_labels[
        323
    ] = "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk"
    imagenet_labels[324] = "damselfly"
    imagenet_labels[325] = "admiral"
    imagenet_labels[326] = "ringlet, ringlet butterfly"
    imagenet_labels[
        327
    ] = "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus"
    imagenet_labels[328] = "cabbage butterfly"
    imagenet_labels[329] = "sulphur butterfly, sulfur butterfly"
    imagenet_labels[330] = "lycaenid, lycaenid butterfly"
    imagenet_labels[331] = "starfish, sea star"
    imagenet_labels[332] = "sea urchin"
    imagenet_labels[333] = "sea cucumber, holothurian"
    imagenet_labels[334] = "wood rabbit, cottontail, cottontail rabbit"
    imagenet_labels[335] = "hare"
    imagenet_labels[336] = "Angora, Angora rabbit"
    imagenet_labels[337] = "hamster"
    imagenet_labels[338] = "porcupine, hedgehog"
    imagenet_labels[339] = "fox squirrel, eastern fox squirrel, Sciurus niger"
    imagenet_labels[340] = "marmot"
    imagenet_labels[341] = "beaver"
    imagenet_labels[342] = "guinea pig, Cavia cobaya"
    imagenet_labels[343] = "sorrel"
    imagenet_labels[344] = "zebra"
    imagenet_labels[345] = "hog, pig, grunter, squealer, Sus scrofa"
    imagenet_labels[346] = "wild boar, boar, Sus scrofa"
    imagenet_labels[347] = "warthog"
    imagenet_labels[348] = "hippopotamus, hippo, Hippopotamus amphibius"
    imagenet_labels[349] = "ox"
    imagenet_labels[350] = "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis"
    imagenet_labels[351] = "bison"
    imagenet_labels[352] = "ram, tup"
    imagenet_labels[
        353
    ] = "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis"
    imagenet_labels[354] = "ibex, Capra ibex"
    imagenet_labels[355] = "hartebeest"
    imagenet_labels[356] = "impala, Aepyceros melampus"
    imagenet_labels[357] = "gazelle"
    imagenet_labels[358] = "Arabian camel, dromedary, Camelus dromedarius"
    imagenet_labels[359] = "llama"
    imagenet_labels[360] = "weasel"
    imagenet_labels[361] = "mink"
    imagenet_labels[362] = "polecat, fitch, foulmart, foumart, Mustela putorius"
    imagenet_labels[363] = "black-footed ferret, ferret, Mustela nigripes"
    imagenet_labels[364] = "otter"
    imagenet_labels[365] = "skunk, polecat, wood pussy"
    imagenet_labels[366] = "badger"
    imagenet_labels[367] = "armadillo"
    imagenet_labels[368] = "three-toed sloth, ai, Bradypus tridactylus"
    imagenet_labels[369] = "orangutan, orang, orangutang, Pongo pygmaeus"
    imagenet_labels[370] = "gorilla, Gorilla gorilla"
    imagenet_labels[371] = "chimpanzee, chimp, Pan troglodytes"
    imagenet_labels[372] = "gibbon, Hylobates lar"
    imagenet_labels[373] = "siamang, Hylobates syndactylus, Symphalangus syndactylus"
    imagenet_labels[374] = "guenon, guenon monkey"
    imagenet_labels[375] = "patas, hussar monkey, Erythrocebus patas"
    imagenet_labels[376] = "baboon"
    imagenet_labels[377] = "macaque"
    imagenet_labels[378] = "langur"
    imagenet_labels[379] = "colobus, colobus monkey"
    imagenet_labels[380] = "proboscis monkey, Nasalis larvatus"
    imagenet_labels[381] = "marmoset"
    imagenet_labels[382] = "capuchin, ringtail, Cebus capucinus"
    imagenet_labels[383] = "howler monkey, howler"
    imagenet_labels[384] = "titi, titi monkey"
    imagenet_labels[385] = "spider monkey, Ateles geoffroyi"
    imagenet_labels[386] = "squirrel monkey, Saimiri sciureus"
    imagenet_labels[387] = "Madagascar cat, ring-tailed lemur, Lemur catta"
    imagenet_labels[388] = "indri, indris, Indri indri, Indri brevicaudatus"
    imagenet_labels[389] = "Indian elephant, Elephas maximus"
    imagenet_labels[390] = "African elephant, Loxodonta africana"
    imagenet_labels[
        391
    ] = "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens"
    imagenet_labels[
        392
    ] = "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca"
    imagenet_labels[393] = "barracouta, snoek"
    imagenet_labels[394] = "eel"
    imagenet_labels[
        395
    ] = "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch"
    imagenet_labels[396] = "rock beauty, Holocanthus tricolor"
    imagenet_labels[397] = "anemone fish"
    imagenet_labels[398] = "sturgeon"
    imagenet_labels[399] = "gar, garfish, garpike, billfish, Lepisosteus osseus"
    imagenet_labels[400] = "lionfish"
    imagenet_labels[401] = "puffer, pufferfish, blowfish, globefish"
    imagenet_labels[402] = "abacus"
    imagenet_labels[403] = "abaya"
    imagenet_labels[404] = "academic gown, academic robe, judge's robe"
    imagenet_labels[405] = "accordion, piano accordion, squeeze box"
    imagenet_labels[406] = "acoustic guitar"
    imagenet_labels[407] = "aircraft carrier, carrier, flattop, attack aircraft carrier"
    imagenet_labels[408] = "airliner"
    imagenet_labels[409] = "airship, dirigible"
    imagenet_labels[410] = "altar"
    imagenet_labels[411] = "ambulance"
    imagenet_labels[412] = "amphibian, amphibious vehicle"
    imagenet_labels[413] = "analog clock"
    imagenet_labels[414] = "apiary, bee house"
    imagenet_labels[415] = "apron"
    imagenet_labels[
        416
    ] = "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin"
    imagenet_labels[417] = "assault rifle, assault gun"
    imagenet_labels[
        418
    ] = "backpack, back pack, knapsack, packsack, rucksack, haversack"
    imagenet_labels[419] = "bakery, bakeshop, bakehouse"
    imagenet_labels[420] = "balance beam, beam"
    imagenet_labels[421] = "balloon"
    imagenet_labels[422] = "ballpoint, ballpoint pen, ballpen, Biro"
    imagenet_labels[423] = "Band Aid"
    imagenet_labels[424] = "banjo"
    imagenet_labels[425] = "bannister, banister, balustrade, balusters, handrail"
    imagenet_labels[426] = "barbell"
    imagenet_labels[427] = "barber chair"
    imagenet_labels[428] = "barbershop"
    imagenet_labels[429] = "barn"
    imagenet_labels[430] = "barometer"
    imagenet_labels[431] = "barrel, cask"
    imagenet_labels[432] = "barrow, garden cart, lawn cart, wheelbarrow"
    imagenet_labels[433] = "baseball"
    imagenet_labels[434] = "basketball"
    imagenet_labels[435] = "bassinet"
    imagenet_labels[436] = "bassoon"
    imagenet_labels[437] = "bathing cap, swimming cap"
    imagenet_labels[438] = "bath towel"
    imagenet_labels[439] = "bathtub, bathing tub, bath, tub"
    imagenet_labels[
        440
    ] = "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon"
    imagenet_labels[441] = "beacon, lighthouse, beacon light, pharos"
    imagenet_labels[442] = "beaker"
    imagenet_labels[443] = "bearskin, busby, shako"
    imagenet_labels[444] = "beer bottle"
    imagenet_labels[445] = "beer glass"
    imagenet_labels[446] = "bell cote, bell cot"
    imagenet_labels[447] = "bib"
    imagenet_labels[448] = "bicycle-built-for-two, tandem bicycle, tandem"
    imagenet_labels[449] = "bikini, two-piece"
    imagenet_labels[450] = "binder, ring-binder"
    imagenet_labels[451] = "binoculars, field glasses, opera glasses"
    imagenet_labels[452] = "birdhouse"
    imagenet_labels[453] = "boathouse"
    imagenet_labels[454] = "bobsled, bobsleigh, bob"
    imagenet_labels[455] = "bolo tie, bolo, bola tie, bola"
    imagenet_labels[456] = "bonnet, poke bonnet"
    imagenet_labels[457] = "bookcase"
    imagenet_labels[458] = "bookshop, bookstore, bookstall"
    imagenet_labels[459] = "bottlecap"
    imagenet_labels[460] = "bow"
    imagenet_labels[461] = "bow tie, bow-tie, bowtie"
    imagenet_labels[462] = "brass, memorial tablet, plaque"
    imagenet_labels[463] = "brassiere, bra, bandeau"
    imagenet_labels[464] = "breakwater, groin, groyne, mole, bulwark, seawall, jetty"
    imagenet_labels[465] = "breastplate, aegis, egis"
    imagenet_labels[466] = "broom"
    imagenet_labels[467] = "bucket, pail"
    imagenet_labels[468] = "buckle"
    imagenet_labels[469] = "bulletproof vest"
    imagenet_labels[470] = "bullet train, bullet"
    imagenet_labels[471] = "butcher shop, meat market"
    imagenet_labels[472] = "cab, hack, taxi, taxicab"
    imagenet_labels[473] = "caldron, cauldron"
    imagenet_labels[474] = "candle, taper, wax light"
    imagenet_labels[475] = "cannon"
    imagenet_labels[476] = "canoe"
    imagenet_labels[477] = "can opener, tin opener"
    imagenet_labels[478] = "cardigan"
    imagenet_labels[479] = "car mirror"
    imagenet_labels[480] = "carousel, carrousel, merry-go-round, roundabout, whirligig"
    imagenet_labels[481] = "carpenter's kit, tool kit"
    imagenet_labels[482] = "carton"
    imagenet_labels[483] = "car wheel"
    imagenet_labels[
        484
    ] = "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM"
    imagenet_labels[485] = "cassette"
    imagenet_labels[486] = "cassette player"
    imagenet_labels[487] = "castle"
    imagenet_labels[488] = "catamaran"
    imagenet_labels[489] = "CD player"
    imagenet_labels[490] = "cello, violoncello"
    imagenet_labels[
        491
    ] = "cellular telephone, cellular phone, cellphone, cell, mobile phone"
    imagenet_labels[492] = "chain"
    imagenet_labels[493] = "chainlink fence"
    imagenet_labels[
        494
    ] = "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour"
    imagenet_labels[495] = "chain saw, chainsaw"
    imagenet_labels[496] = "chest"
    imagenet_labels[497] = "chiffonier, commode"
    imagenet_labels[498] = "chime, bell, gong"
    imagenet_labels[499] = "china cabinet, china closet"
    imagenet_labels[500] = "Christmas stocking"
    imagenet_labels[501] = "church, church building"
    imagenet_labels[
        502
    ] = "cinema, movie theater, movie theatre, movie house, picture palace"
    imagenet_labels[503] = "cleaver, meat cleaver, chopper"
    imagenet_labels[504] = "cliff dwelling"
    imagenet_labels[505] = "cloak"
    imagenet_labels[506] = "clog, geta, patten, sabot"
    imagenet_labels[507] = "cocktail shaker"
    imagenet_labels[508] = "coffee mug"
    imagenet_labels[509] = "coffeepot"
    imagenet_labels[510] = "coil, spiral, volute, whorl, helix"
    imagenet_labels[511] = "combination lock"
    imagenet_labels[512] = "computer keyboard, keypad"
    imagenet_labels[513] = "confectionery, confectionary, candy store"
    imagenet_labels[514] = "container ship, containership, container vessel"
    imagenet_labels[515] = "convertible"
    imagenet_labels[516] = "corkscrew, bottle screw"
    imagenet_labels[517] = "cornet, horn, trumpet, trump"
    imagenet_labels[518] = "cowboy boot"
    imagenet_labels[519] = "cowboy hat, ten-gallon hat"
    imagenet_labels[520] = "cradle"
    imagenet_labels[521] = "crane"
    imagenet_labels[522] = "crash helmet"
    imagenet_labels[523] = "crate"
    imagenet_labels[524] = "crib, cot"
    imagenet_labels[525] = "Crock Pot"
    imagenet_labels[526] = "croquet ball"
    imagenet_labels[527] = "crutch"
    imagenet_labels[528] = "cuirass"
    imagenet_labels[529] = "dam, dike, dyke"
    imagenet_labels[530] = "desk"
    imagenet_labels[531] = "desktop computer"
    imagenet_labels[532] = "dial telephone, dial phone"
    imagenet_labels[533] = "diaper, nappy, napkin"
    imagenet_labels[534] = "digital clock"
    imagenet_labels[535] = "digital watch"
    imagenet_labels[536] = "dining table, board"
    imagenet_labels[537] = "dishrag, dishcloth"
    imagenet_labels[538] = "dishwasher, dish washer, dishwashing machine"
    imagenet_labels[539] = "disk brake, disc brake"
    imagenet_labels[540] = "dock, dockage, docking facility"
    imagenet_labels[541] = "dogsled, dog sled, dog sleigh"
    imagenet_labels[542] = "dome"
    imagenet_labels[543] = "doormat, welcome mat"
    imagenet_labels[544] = "drilling platform, offshore rig"
    imagenet_labels[545] = "drum, membranophone, tympan"
    imagenet_labels[546] = "drumstick"
    imagenet_labels[547] = "dumbbell"
    imagenet_labels[548] = "Dutch oven"
    imagenet_labels[549] = "electric fan, blower"
    imagenet_labels[550] = "electric guitar"
    imagenet_labels[551] = "electric locomotive"
    imagenet_labels[552] = "entertainment center"
    imagenet_labels[553] = "envelope"
    imagenet_labels[554] = "espresso maker"
    imagenet_labels[555] = "face powder"
    imagenet_labels[556] = "feather boa, boa"
    imagenet_labels[557] = "file, file cabinet, filing cabinet"
    imagenet_labels[558] = "fireboat"
    imagenet_labels[559] = "fire engine, fire truck"
    imagenet_labels[560] = "fire screen, fireguard"
    imagenet_labels[561] = "flagpole, flagstaff"
    imagenet_labels[562] = "flute, transverse flute"
    imagenet_labels[563] = "folding chair"
    imagenet_labels[564] = "football helmet"
    imagenet_labels[565] = "forklift"
    imagenet_labels[566] = "fountain"
    imagenet_labels[567] = "fountain pen"
    imagenet_labels[568] = "four-poster"
    imagenet_labels[569] = "freight car"
    imagenet_labels[570] = "French horn, horn"
    imagenet_labels[571] = "frying pan, frypan, skillet"
    imagenet_labels[572] = "fur coat"
    imagenet_labels[573] = "garbage truck, dustcart"
    imagenet_labels[574] = "gasmask, respirator, gas helmet"
    imagenet_labels[575] = "gas pump, gasoline pump, petrol pump, island dispenser"
    imagenet_labels[576] = "goblet"
    imagenet_labels[577] = "go-kart"
    imagenet_labels[578] = "golf ball"
    imagenet_labels[579] = "golfcart, golf cart"
    imagenet_labels[580] = "gondola"
    imagenet_labels[581] = "gong, tam-tam"
    imagenet_labels[582] = "gown"
    imagenet_labels[583] = "grand piano, grand"
    imagenet_labels[584] = "greenhouse, nursery, glasshouse"
    imagenet_labels[585] = "grille, radiator grille"
    imagenet_labels[586] = "grocery store, grocery, food market, market"
    imagenet_labels[587] = "guillotine"
    imagenet_labels[588] = "hair slide"
    imagenet_labels[589] = "hair spray"
    imagenet_labels[590] = "half track"
    imagenet_labels[591] = "hammer"
    imagenet_labels[592] = "hamper"
    imagenet_labels[593] = "hand blower, blow dryer, blow drier, hair dryer, hair drier"
    imagenet_labels[594] = "hand-held computer, hand-held microcomputer"
    imagenet_labels[595] = "handkerchief, hankie, hanky, hankey"
    imagenet_labels[596] = "hard disk, hard disc, fixed disk"
    imagenet_labels[597] = "harmonica, mouth organ, harp, mouth harp"
    imagenet_labels[598] = "harp"
    imagenet_labels[599] = "hatchet"
    imagenet_labels[600] = "holster"
    imagenet_labels[601] = "home theater, home theatre"
    imagenet_labels[602] = "honeycomb"
    imagenet_labels[603] = "hook, claw"
    imagenet_labels[604] = "hoopskirt, crinoline"
    imagenet_labels[605] = "horizontal bar, high bar"
    imagenet_labels[606] = "horse cart, horse-cart"
    imagenet_labels[607] = "hourglass"
    imagenet_labels[608] = "iPod"
    imagenet_labels[609] = "iron, smoothing iron"
    imagenet_labels[610] = "jack-o'-lantern"
    imagenet_labels[611] = "jean, blue jean, denim"
    imagenet_labels[612] = "jeep, landrover"
    imagenet_labels[613] = "jersey, T-shirt, tee shirt"
    imagenet_labels[614] = "jigsaw puzzle"
    imagenet_labels[615] = "jinrikisha, ricksha, rickshaw"
    imagenet_labels[616] = "joystick"
    imagenet_labels[617] = "kimono"
    imagenet_labels[618] = "knee pad"
    imagenet_labels[619] = "knot"
    imagenet_labels[620] = "lab coat, laboratory coat"
    imagenet_labels[621] = "ladle"
    imagenet_labels[622] = "lampshade, lamp shade"
    imagenet_labels[623] = "laptop, laptop computer"
    imagenet_labels[624] = "lawn mower, mower"
    imagenet_labels[625] = "lens cap, lens cover"
    imagenet_labels[626] = "letter opener, paper knife, paperknife"
    imagenet_labels[627] = "library"
    imagenet_labels[628] = "lifeboat"
    imagenet_labels[629] = "lighter, light, igniter, ignitor"
    imagenet_labels[630] = "limousine, limo"
    imagenet_labels[631] = "liner, ocean liner"
    imagenet_labels[632] = "lipstick, lip rouge"
    imagenet_labels[633] = "Loafer"
    imagenet_labels[634] = "lotion"
    imagenet_labels[
        635
    ] = "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system"
    imagenet_labels[636] = "loupe, jeweler's loupe"
    imagenet_labels[637] = "lumbermill, sawmill"
    imagenet_labels[638] = "magnetic compass"
    imagenet_labels[639] = "mailbag, postbag"
    imagenet_labels[640] = "mailbox, letter box"
    imagenet_labels[641] = "maillot"
    imagenet_labels[642] = "maillot, tank suit"
    imagenet_labels[643] = "manhole cover"
    imagenet_labels[644] = "maraca"
    imagenet_labels[645] = "marimba, xylophone"
    imagenet_labels[646] = "mask"
    imagenet_labels[647] = "matchstick"
    imagenet_labels[648] = "maypole"
    imagenet_labels[649] = "maze, labyrinth"
    imagenet_labels[650] = "measuring cup"
    imagenet_labels[651] = "medicine chest, medicine cabinet"
    imagenet_labels[652] = "megalith, megalithic structure"
    imagenet_labels[653] = "microphone, mike"
    imagenet_labels[654] = "microwave, microwave oven"
    imagenet_labels[655] = "military uniform"
    imagenet_labels[656] = "milk can"
    imagenet_labels[657] = "minibus"
    imagenet_labels[658] = "miniskirt, mini"
    imagenet_labels[659] = "minivan"
    imagenet_labels[660] = "missile"
    imagenet_labels[661] = "mitten"
    imagenet_labels[662] = "mixing bowl"
    imagenet_labels[663] = "mobile home, manufactured home"
    imagenet_labels[664] = "Model T"
    imagenet_labels[665] = "modem"
    imagenet_labels[666] = "monastery"
    imagenet_labels[667] = "monitor"
    imagenet_labels[668] = "moped"
    imagenet_labels[669] = "mortar"
    imagenet_labels[670] = "mortarboard"
    imagenet_labels[671] = "mosque"
    imagenet_labels[672] = "mosquito net"
    imagenet_labels[673] = "motor scooter, scooter"
    imagenet_labels[674] = "mountain bike, all-terrain bike, off-roader"
    imagenet_labels[675] = "mountain tent"
    imagenet_labels[676] = "mouse, computer mouse"
    imagenet_labels[677] = "mousetrap"
    imagenet_labels[678] = "moving van"
    imagenet_labels[679] = "muzzle"
    imagenet_labels[680] = "nail"
    imagenet_labels[681] = "neck brace"
    imagenet_labels[682] = "necklace"
    imagenet_labels[683] = "nipple"
    imagenet_labels[684] = "notebook, notebook computer"
    imagenet_labels[685] = "obelisk"
    imagenet_labels[686] = "oboe, hautboy, hautbois"
    imagenet_labels[687] = "ocarina, sweet potato"
    imagenet_labels[688] = "odometer, hodometer, mileometer, milometer"
    imagenet_labels[689] = "oil filter"
    imagenet_labels[690] = "organ, pipe organ"
    imagenet_labels[691] = "oscilloscope, scope, cathode-ray oscilloscope, CRO"
    imagenet_labels[692] = "overskirt"
    imagenet_labels[693] = "oxcart"
    imagenet_labels[694] = "oxygen mask"
    imagenet_labels[695] = "packet"
    imagenet_labels[696] = "paddle, boat paddle"
    imagenet_labels[697] = "paddlewheel, paddle wheel"
    imagenet_labels[698] = "padlock"
    imagenet_labels[699] = "paintbrush"
    imagenet_labels[700] = "pajamas, pyjamas, pj's, jammies"
    imagenet_labels[701] = "palace"
    imagenet_labels[702] = "panpipe, pandean pipe, syrinx"
    imagenet_labels[703] = "paper towel"
    imagenet_labels[704] = "parachute, chute"
    imagenet_labels[705] = "parallel bars, bars"
    imagenet_labels[706] = "park bench"
    imagenet_labels[707] = "parking meter"
    imagenet_labels[708] = "passenger car, coach, carriage"
    imagenet_labels[709] = "patio, terrace"
    imagenet_labels[710] = "pay-phone, pay-station"
    imagenet_labels[711] = "pedestal, plinth, footstall"
    imagenet_labels[712] = "pencil box, pencil case"
    imagenet_labels[713] = "pencil sharpener"
    imagenet_labels[714] = "perfume, essence"
    imagenet_labels[715] = "Petri dish"
    imagenet_labels[716] = "photocopier"
    imagenet_labels[717] = "pick, plectrum, plectron"
    imagenet_labels[718] = "pickelhaube"
    imagenet_labels[719] = "picket fence, paling"
    imagenet_labels[720] = "pickup, pickup truck"
    imagenet_labels[721] = "pier"
    imagenet_labels[722] = "piggy bank, penny bank"
    imagenet_labels[723] = "pill bottle"
    imagenet_labels[724] = "pillow"
    imagenet_labels[725] = "ping-pong ball"
    imagenet_labels[726] = "pinwheel"
    imagenet_labels[727] = "pirate, pirate ship"
    imagenet_labels[728] = "pitcher, ewer"
    imagenet_labels[729] = "plane, carpenter's plane, woodworking plane"
    imagenet_labels[730] = "planetarium"
    imagenet_labels[731] = "plastic bag"
    imagenet_labels[732] = "plate rack"
    imagenet_labels[733] = "plow, plough"
    imagenet_labels[734] = "plunger, plumber's helper"
    imagenet_labels[735] = "Polaroid camera, Polaroid Land camera"
    imagenet_labels[736] = "pole"
    imagenet_labels[
        737
    ] = "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria"
    imagenet_labels[738] = "poncho"
    imagenet_labels[739] = "pool table, billiard table, snooker table"
    imagenet_labels[740] = "pop bottle, soda bottle"
    imagenet_labels[741] = "pot, flowerpot"
    imagenet_labels[742] = "potter's wheel"
    imagenet_labels[743] = "power drill"
    imagenet_labels[744] = "prayer rug, prayer mat"
    imagenet_labels[745] = "printer"
    imagenet_labels[746] = "prison, prison house"
    imagenet_labels[747] = "projectile, missile"
    imagenet_labels[748] = "projector"
    imagenet_labels[749] = "puck, hockey puck"
    imagenet_labels[750] = "punching bag, punch bag, punching ball, punchball"
    imagenet_labels[751] = "purse"
    imagenet_labels[752] = "quill, quill pen"
    imagenet_labels[753] = "quilt, comforter, comfort, puff"
    imagenet_labels[754] = "racer, race car, racing car"
    imagenet_labels[755] = "racket, racquet"
    imagenet_labels[756] = "radiator"
    imagenet_labels[757] = "radio, wireless"
    imagenet_labels[758] = "radio telescope, radio reflector"
    imagenet_labels[759] = "rain barrel"
    imagenet_labels[760] = "raincoat"
    imagenet_labels[761] = "recreational vehicle, RV, R.V."
    imagenet_labels[762] = "reel"
    imagenet_labels[763] = "reflex camera"
    imagenet_labels[764] = "refrigerator, icebox"
    imagenet_labels[765] = "remote control, remote"
    imagenet_labels[766] = "restaurant, eating house, eating place, eatery"
    imagenet_labels[767] = "revolver, six-gun, six-shooter"
    imagenet_labels[768] = "rifle"
    imagenet_labels[769] = "rocking chair, rocker"
    imagenet_labels[770] = "rotisserie"
    imagenet_labels[771] = "rubber eraser, rubber, pencil eraser"
    imagenet_labels[772] = "rugby ball"
    imagenet_labels[773] = "rule, ruler"
    imagenet_labels[774] = "running shoe"
    imagenet_labels[775] = "safe"
    imagenet_labels[776] = "safety pin"
    imagenet_labels[777] = "saltshaker, salt shaker"
    imagenet_labels[778] = "sandal"
    imagenet_labels[779] = "sarong"
    imagenet_labels[780] = "saxophone, sax"
    imagenet_labels[781] = "scabbard"
    imagenet_labels[782] = "scale, weighing machine"
    imagenet_labels[783] = "school bus"
    imagenet_labels[784] = "schooner"
    imagenet_labels[785] = "scoreboard"
    imagenet_labels[786] = "screen, CRT screen"
    imagenet_labels[787] = "screw"
    imagenet_labels[788] = "screwdriver"
    imagenet_labels[789] = "seat belt, seatbelt"
    imagenet_labels[790] = "sewing machine"
    imagenet_labels[791] = "shield, buckler"
    imagenet_labels[792] = "shoe shop, shoe-shop, shoe store"
    imagenet_labels[793] = "shoji"
    imagenet_labels[794] = "shopping basket"
    imagenet_labels[795] = "shopping cart"
    imagenet_labels[796] = "shovel"
    imagenet_labels[797] = "shower cap"
    imagenet_labels[798] = "shower curtain"
    imagenet_labels[799] = "ski"
    imagenet_labels[800] = "ski mask"
    imagenet_labels[801] = "sleeping bag"
    imagenet_labels[802] = "slide rule, slipstick"
    imagenet_labels[803] = "sliding door"
    imagenet_labels[804] = "slot, one-armed bandit"
    imagenet_labels[805] = "snorkel"
    imagenet_labels[806] = "snowmobile"
    imagenet_labels[807] = "snowplow, snowplough"
    imagenet_labels[808] = "soap dispenser"
    imagenet_labels[809] = "soccer ball"
    imagenet_labels[810] = "sock"
    imagenet_labels[811] = "solar dish, solar collector, solar furnace"
    imagenet_labels[812] = "sombrero"
    imagenet_labels[813] = "soup bowl"
    imagenet_labels[814] = "space bar"
    imagenet_labels[815] = "space heater"
    imagenet_labels[816] = "space shuttle"
    imagenet_labels[817] = "spatula"
    imagenet_labels[818] = "speedboat"
    imagenet_labels[819] = "spider web, spider's web"
    imagenet_labels[820] = "spindle"
    imagenet_labels[821] = "sports car, sport car"
    imagenet_labels[822] = "spotlight, spot"
    imagenet_labels[823] = "stage"
    imagenet_labels[824] = "steam locomotive"
    imagenet_labels[825] = "steel arch bridge"
    imagenet_labels[826] = "steel drum"
    imagenet_labels[827] = "stethoscope"
    imagenet_labels[828] = "stirrup"
    imagenet_labels[829] = "stole"
    imagenet_labels[830] = "stone wall"
    imagenet_labels[831] = "stopwatch, stop watch"
    imagenet_labels[832] = "stove"
    imagenet_labels[833] = "strainer"
    imagenet_labels[834] = "streetcar, tram, tramcar, trolley, trolley car"
    imagenet_labels[835] = "stretcher"
    imagenet_labels[836] = "studio couch, day bed"
    imagenet_labels[837] = "stupa, tope"
    imagenet_labels[838] = "submarine, sub, U-boat"
    imagenet_labels[839] = "suit, suit of clothes"
    imagenet_labels[840] = "sundial"
    imagenet_labels[841] = "sunglass"
    imagenet_labels[842] = "sunglasses, dark glasses, shades"
    imagenet_labels[843] = "sunscreen, sunblock, sun blocker"
    imagenet_labels[844] = "suspension bridge"
    imagenet_labels[845] = "swab, swob"
    imagenet_labels[846] = "sweatshirt"
    imagenet_labels[847] = "swimming trunks, bathing trunks"
    imagenet_labels[848] = "swing"
    imagenet_labels[849] = "switch, electric switch, electrical switch"
    imagenet_labels[850] = "syringe"
    imagenet_labels[851] = "table lamp"
    imagenet_labels[
        852
    ] = "tank, army tank, armored combat vehicle, armoured combat vehicle"
    imagenet_labels[853] = "tape player"
    imagenet_labels[854] = "teapot"
    imagenet_labels[855] = "teddy, teddy bear"
    imagenet_labels[856] = "television, television system"
    imagenet_labels[857] = "tennis ball"
    imagenet_labels[858] = "thatch, thatched roof"
    imagenet_labels[859] = "theater curtain, theatre curtain"
    imagenet_labels[860] = "thimble"
    imagenet_labels[861] = "thresher, threshing machine"
    imagenet_labels[862] = "throne"
    imagenet_labels[863] = "tile roof"
    imagenet_labels[864] = "toaster"
    imagenet_labels[865] = "tobacco shop, tobacconist shop, tobacconist"
    imagenet_labels[866] = "toilet seat"
    imagenet_labels[867] = "torch"
    imagenet_labels[868] = "totem pole"
    imagenet_labels[869] = "tow truck, tow car, wrecker"
    imagenet_labels[870] = "toyshop"
    imagenet_labels[871] = "tractor"
    imagenet_labels[
        872
    ] = "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi"
    imagenet_labels[873] = "tram, tramcar, streetcar, trolley, trolley car"
    imagenet_labels[874] = "trampoline"
    imagenet_labels[875] = "trash can, trash bin, garbage can, garbage bin, wastebin"
    imagenet_labels[876] = "treadmill"
    imagenet_labels[877] = "trench coat"
    imagenet_labels[878] = "tricycle, trike, velocipede"
    imagenet_labels[879] = "trimaran"
    imagenet_labels[880] = "tripod"
    imagenet_labels[881] = "triumphal arch"
    imagenet_labels[882] = "trolleybus, trolley coach, trackless trolley"
    imagenet_labels[883] = "trombone"
    imagenet_labels[884] = "tub, vat"
    imagenet_labels[885] = "turnstile"
    imagenet_labels[886] = "typewriter keyboard"
    imagenet_labels[887] = "umbrella"
    imagenet_labels[888] = "unicycle, monocycle"
    imagenet_labels[889] = "upright, upright piano"
    imagenet_labels[890] = "vacuum, vacuum cleaner"
    imagenet_labels[891] = "vase"
    imagenet_labels[892] = "vault"
    imagenet_labels[893] = "velvet"
    imagenet_labels[894] = "vending machine"
    imagenet_labels[895] = "vestment"
    imagenet_labels[896] = "viaduct"
    imagenet_labels[897] = "violin, fiddle"
    imagenet_labels[898] = "volleyball"
    imagenet_labels[899] = "waffle iron"
    imagenet_labels[900] = "wall clock"
    imagenet_labels[901] = "wallet, billfold, notecase, pocketbook"
    imagenet_labels[902] = "wardrobe, closet, press"
    imagenet_labels[903] = "warplane, military plane"
    imagenet_labels[904] = "washbasin, handbasin, washbowl, lavabo, wash-hand basin"
    imagenet_labels[905] = "washer, automatic washer, washing machine"
    imagenet_labels[906] = "water bottle"
    imagenet_labels[907] = "water jug"
    imagenet_labels[908] = "water tower"
    imagenet_labels[909] = "whiskey jug"
    imagenet_labels[910] = "whistle"
    imagenet_labels[911] = "wig"
    imagenet_labels[912] = "window screen"
    imagenet_labels[913] = "window shade"
    imagenet_labels[914] = "Windsor tie"
    imagenet_labels[915] = "wine bottle"
    imagenet_labels[916] = "wine glass"
    imagenet_labels[917] = "wireless telephone, radiotelephone, radiophone"
    imagenet_labels[918] = "witch hat"
    imagenet_labels[919] = "wok"
    imagenet_labels[920] = "wooden spoon"
    imagenet_labels[921] = "wool, woolen, woollen"
    imagenet_labels[922] = "worm fence, snake fence, snake-rail fence, Virginia fence"
    imagenet_labels[923] = "wreck"
    imagenet_labels[924] = "yawl"
    imagenet_labels[925] = "yurt"
    imagenet_labels[926] = "web site, website, internet site, site"
    imagenet_labels[927] = "comic book"
    imagenet_labels[928] = "crossword puzzle, crossword"
    imagenet_labels[929] = "street sign"
    imagenet_labels[930] = "traffic light, traffic signal, stoplight"
    imagenet_labels[931] = "book jacket, dust cover, dust jacket, dust wrapper"
    imagenet_labels[932] = "menu"
    imagenet_labels[933] = "plate"
    imagenet_labels[934] = "guacamole"
    imagenet_labels[935] = "consomme"
    imagenet_labels[936] = "hot pot, hotpot"
    imagenet_labels[937] = "trifle"
    imagenet_labels[938] = "ice cream, icecream"
    imagenet_labels[939] = "ice lolly, lolly, lollipop, popsicle"
    imagenet_labels[940] = "French loaf"
    imagenet_labels[941] = "bagel, beigel"
    imagenet_labels[942] = "pretzel"
    imagenet_labels[943] = "cheeseburger"
    imagenet_labels[944] = "hotdog, hot dog, red hot"
    imagenet_labels[945] = "mashed potato"
    imagenet_labels[946] = "head cabbage"
    imagenet_labels[947] = "broccoli"
    imagenet_labels[948] = "cauliflower"
    imagenet_labels[949] = "zucchini, courgette"
    imagenet_labels[950] = "spaghetti squash"
    imagenet_labels[951] = "acorn squash"
    imagenet_labels[952] = "butternut squash"
    imagenet_labels[953] = "cucumber, cuke"
    imagenet_labels[954] = "artichoke, globe artichoke"
    imagenet_labels[955] = "bell pepper"
    imagenet_labels[956] = "cardoon"
    imagenet_labels[957] = "mushroom"
    imagenet_labels[958] = "Granny Smith"
    imagenet_labels[959] = "strawberry"
    imagenet_labels[960] = "orange"
    imagenet_labels[961] = "lemon"
    imagenet_labels[962] = "fig"
    imagenet_labels[963] = "pineapple, ananas"
    imagenet_labels[964] = "banana"
    imagenet_labels[965] = "jackfruit, jak, jack"
    imagenet_labels[966] = "custard apple"
    imagenet_labels[967] = "pomegranate"
    imagenet_labels[968] = "hay"
    imagenet_labels[969] = "carbonara"
    imagenet_labels[970] = "chocolate sauce, chocolate syrup"
    imagenet_labels[971] = "dough"
    imagenet_labels[972] = "meat loaf, meatloaf"
    imagenet_labels[973] = "pizza, pizza pie"
    imagenet_labels[974] = "potpie"
    imagenet_labels[975] = "burrito"
    imagenet_labels[976] = "red wine"
    imagenet_labels[977] = "espresso"
    imagenet_labels[978] = "cup"
    imagenet_labels[979] = "eggnog"
    imagenet_labels[980] = "alp"
    imagenet_labels[981] = "bubble"
    imagenet_labels[982] = "cliff, drop, drop-off"
    imagenet_labels[983] = "coral reef"
    imagenet_labels[984] = "geyser"
    imagenet_labels[985] = "lakeside, lakeshore"
    imagenet_labels[986] = "promontory, headland, head, foreland"
    imagenet_labels[987] = "sandbar, sand bar"
    imagenet_labels[988] = "seashore, coast, seacoast, sea-coast"
    imagenet_labels[989] = "valley, vale"
    imagenet_labels[990] = "volcano"
    imagenet_labels[991] = "ballplayer, baseball player"
    imagenet_labels[992] = "groom, bridegroom"
    imagenet_labels[993] = "scuba diver"
    imagenet_labels[994] = "rapeseed"
    imagenet_labels[995] = "daisy"
    imagenet_labels[
        996
    ] = "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus"
    imagenet_labels[997] = "corn"
    imagenet_labels[998] = "acorn"
    imagenet_labels[999] = "hip, rose hip, rose haw, hawthorn berry"
    return imagenet_labels


def process_image(image_path, model, transform, device, imagenet_labels):
    """处理单张图片并输出结果"""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with paddle.no_grad():
            output = model(img_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Output min: {output._min().item()}, max: {output._max().item()}")
        pred_idx = paddle.argmax(output, dim=1).item()
        print(f"Predicted index: {pred_idx}")
        if pred_idx in imagenet_labels:
            pred_label = imagenet_labels[pred_idx]
        else:
            pred_label = f"其他类别（索引：{pred_idx}）"
        print("=" * 60)
        print(f"📸 测试图片：{image_path}")
        print(f"🎯 预测类别索引：{pred_idx}")
        print(f"🗂️  预测类别名称：{pred_label}")
        print("=" * 60)
        return pred_idx, pred_label
    except Exception as e:
        print(f"❌ 处理图片时出错：{image_path}")
        print(f"错误信息：{str(e)}")
        print("=" * 60)
        return None, None


def get_true_label_from_folder(folder_name):
    """从文件夹名称获取真实标签（支持数字ID和名称匹配）"""
    imagenet_labels = get_imagenet_labels()
    try:
        class_id = int(folder_name)
        if class_id in imagenet_labels:
            return class_id, imagenet_labels[class_id]
    except ValueError:
        pass
    folder_normalized = folder_name.lower().replace("_", " ").replace("-", " ")
    for idx, label in imagenet_labels.items():
        label_normalized = label.lower().replace("_", " ").replace("-", " ")
        if folder_normalized == label_normalized:
            return idx, label
        if (
            folder_normalized in label_normalized
            or label_normalized in folder_normalized
        ):
            return idx, label
        folder_words = set(folder_normalized.split())
        label_words = set(label_normalized.split())
        if folder_words & label_words:
            return idx, label
    return None, None


def main():
    """主函数"""
    print("Hiera模型程序开始运行...")
    model, device = load_model()
    transform = get_transforms()
    imagenet_labels = get_imagenet_labels()
    results = []
    total_images = 0
    correct_predictions = 0
    database_dir = "/media/user/40da1a25-a924-43a2-b274-e33d39ea8680/cv/lzy/hiera-main/hiera-main/examples/database"
    class_stats = {}
    if os.path.exists(database_dir):
        print(f"Processing images in {database_dir}...")
        for class_dir in os.listdir(database_dir):
            class_path = os.path.join(database_dir, class_dir)
            if os.path.isdir(class_path):
                print(f"\nProcessing class: {class_dir}")
                true_idx, true_label = get_true_label_from_folder(class_dir)
                if true_idx is not None:
                    if true_idx not in class_stats:
                        class_stats[true_idx] = {
                            "correct": 0,
                            "total": 0,
                            "name": true_label,
                        }
                for img_file in tqdm(os.listdir(class_path)):
                    if img_file.endswith((".jpeg", ".jpg", ".png")):
                        img_path = os.path.join(class_path, img_file)
                        pred_idx, pred_label = process_image(
                            img_path, model, transform, device, imagenet_labels
                        )
                        if pred_idx is not None and pred_label is not None:
                            total_images += 1
                            is_correct = true_idx is not None and pred_idx == true_idx
                            if is_correct:
                                correct_predictions += 1
                                if true_idx is not None:
                                    class_stats[true_idx]["correct"] += 1
                            if true_idx is not None:
                                class_stats[true_idx]["total"] += 1
                            results.append(
                                {
                                    "图片路径": img_path,
                                    "真实类别文件夹": class_dir,
                                    "真实类别索引": true_idx
                                    if true_idx is not None
                                    else "未知",
                                    "真实类别名称": true_label
                                    if true_label is not None
                                    else "未知",
                                    "预测类别索引": pred_idx,
                                    "预测类别名称": pred_label,
                                    "是否正确": "是" if is_correct else "否",
                                }
                            )
    else:
        print(f"Database directory {database_dir} not found!")
    if total_images > 0:
        accuracy = 100.0 * correct_predictions / total_images
        print(f"\n" + "=" * 60)
        print(f"📊 测试结果")
        print("=" * 60)
        print(f"✅ 总体准确率: {accuracy:.2f}%")
        print(f"   正确预测: {correct_predictions}/{total_images}")
        print(f"   测试图像: {total_images} 张")
        print(f"   测试类别: {len(class_stats)} 个")
    if class_stats:
        print(f"\n📈 各类别准确率:")
        print("-" * 60)
        print(f"{'类别':<15} {'正确/总数':<15} {'准确率':<10}")
        print("-" * 50)
        for class_idx in sorted(class_stats.keys()):
            stats = class_stats[class_idx]
            if stats["total"] > 0:
                class_acc = 100.0 * stats["correct"] / stats["total"]
                class_name = stats["name"]
                print(
                    f"{class_name:<15} {stats['correct']}/{stats['total']:<10} {class_acc:>6.2f}%"
                )
    if results:
        print("\nGenerating Excel report...")
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
        df = pd.DataFrame(results)
        accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
        overall_stats = {
            "统计项": ["测试样本总数", "正确预测数", "错误预测数", "准确率(%)", "测试类别数"],
            "数值": [
                total_images,
                correct_predictions,
                total_images - correct_predictions,
                f"{accuracy:.2f}",
                len(class_stats),
            ],
        }
        df_overall_stats = pd.DataFrame(overall_stats)
        class_stats_list = []
        for class_idx in sorted(class_stats.keys()):
            stats = class_stats[class_idx]
            if stats["total"] > 0:
                class_acc = 100.0 * stats["correct"] / stats["total"]
                class_stats_list.append(
                    {
                        "类别ID": class_idx,
                        "类别名称": stats["name"],
                        "正确预测数": stats["correct"],
                        "总样本数": stats["total"],
                        "错误预测数": stats["total"] - stats["correct"],
                        "准确率(%)": f"{class_acc:.2f}",
                    }
                )
        df_class_stats = pd.DataFrame(class_stats_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(
            data_dir, f"hiera_prediction_results_{timestamp}.xlsx"
        )
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="详细结果", index=False)
            df_overall_stats.to_excel(writer, sheet_name="总体统计", index=False)
            df_class_stats.to_excel(writer, sheet_name="类别统计", index=False)
        print(f"✅ Excel报告生成成功: {excel_path}")
        print(f"\n📊 模型准确率统计:")
        print(f"   - 测试样本总数: {total_images}")
        print(f"   - 正确预测数: {correct_predictions}")
        print(f"   - 错误预测数: {total_images - correct_predictions}")
        print(f"   - 准确率: {accuracy:.2f}%")
        print(f"   - 测试类别数: {len(class_stats)}")
    else:
        print("\nNo results to generate Excel report.")
    print("\n🎉 Hiera模型按官方标准运行成功！")


if __name__ == "__main__":
    main()
