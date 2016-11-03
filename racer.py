# eric.vinck@gmail.com

# keys :
#       space : pause / unpause
#       g : stop capturing, make a CNN model, reset the car to the original position and drive the car using the model
#       r : reset to start (inc. deleting captured dataset)

# todo :
# change the make_vision_cone alg and suppress cos/sin usage, and "black" points
# tune the neural network

import os, sys, pygame, time, numpy
from math import cos,sin,radians,sqrt
from pygame.locals import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def create_help_box():
    text = list()
    text.append("Keys : ")
    text.append("   space : pause / unpause")
    text.append("       g : generate model and use it")
    text.append("       r : reset everything")
    helpbox = pygame.Surface((300,100))
    helpbox.fill((150,150,150))
    font = pygame.font.SysFont("Arial", 15)

    for i,line in enumerate(text):
        helpbox.blit(font.render(line, 1, (0, 0, 0)), (10, i*15))

    return helpbox

def load_image(name, colorkey=None):
    fullname = os.path.join(name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error, message:
        print 'Cannot load image:', name
        raise SystemExit, message
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()

#classes for our game objects
class Car(pygame.sprite.Sprite):

    def __init__(self,startx,starty,startdirection,speedx,speedy):
        pygame.sprite.Sprite.__init__(self) #call Sprite initializer
        self.image, self.rect = load_image('car-small.png', -1)
        self.refimage = self.image
        # the car direction, i.e. where it's heading (in degree : [0-360])
        self.direction = startdirection
        self.posx = startx
        self.posy = starty
        self.rect.centerx = self.posx
        self.rect.centery = self.posy
        self.speedx=speedx
        self.speedy=speedy

    def update(self,timespent):
        #print "direction =",self.direction
        #print " cos(math.radians(self.direction) =",cos(math.radians(self.direction))
        self.posx += self.speedx * cos(radians(self.direction)) * timespent
        self.posy += self.speedy * sin(radians(self.direction)) * timespent

        self.image = pygame.transform.rotate(self.refimage,-self.direction)
        self.rect = self.image.get_rect()

        self.rect.centerx = self.posx
        self.rect.centery = self.posy

    def updateDirection(self, change):
        self.direction += change;
        self.direction = self.direction % 360

def make_vision_cone(coneAngle,coneLength):
    width = sin(radians(coneAngle/2)) * 2 * coneLength
    visionCone = pygame.Surface((coneLength, width))
    visionCone.set_colorkey(0)
    visionCone.set_alpha(200)

    pygame.draw.line(visionCone,(0,255,255),(0,width/2),(coneLength*cos(radians(coneAngle)/2),0),1)
    pygame.draw.line(visionCone,(0,255,255),(0,width/2),(coneLength*cos(radians(coneAngle)/2),width),1)

    j = radians(-coneAngle / 2)
    while j < radians(coneAngle / 2):
        tox = round(cos(j) * (coneLength-1))
        toy = round(sin(j) * (coneLength-1) + width / 2)
        tox = int(tox)
        toy = int(toy)
        visionCone.set_at((tox, toy), (0, 255, 255))
        j += 0.01

    # I set a red dot at the basis of the cone for
    visionCone.set_at((0,int(width/2)),(255,0,0))
    visionCone.set_at((1, int(width / 2)), (255, 0, 0))
    visionCone.set_at((1, int(width / 2)+1), (255, 0, 0))
    visionCone.set_at((1, int(width / 2)-1), (255, 0, 0))

    return visionCone

def cached_cosinus(r):
    pass

def get_vision_cone(display,x,y,direction, coneAngle, coneLength):
    width = sin(radians(coneAngle/2)) * 2 * coneLength
    visionCone = pygame.Surface((coneLength, width))
    visionCone.set_colorkey(0)

    # important : display.get_at and display.set_at are slow
    # so I will use pixelarray (cf pygame documentation)
    pixelArrayDisplay = pygame.PixelArray(display)
    pixelArrayVisioneCone = pygame.PixelArray(visionCone)

    l=0
    for i in range(0,coneLength,1):
        # nb we do the loop in radians because degrees are too blunt for drawing big cones
        j = radians(-coneAngle/2)
        while j<radians(coneAngle/2):
            tox = cos(j)*i
            toy = sin(j)*i+width/2
            tox = int(tox)
            toy = int(toy)

            k = j + radians(direction)
            atx = cos(k)*i
            aty = sin(k)*i
            atx = int(atx + x)
            aty = int(aty + y)

            if (atx <0): atx=0
            atx = min(atx,display.get_width()-1)
            if (aty <0): aty=0
            aty = min(aty,display.get_height()-1)

            #visionCone.set_at((tox,toy),display.get_at((atx,aty)))
            #print "tox=%d toy=%d atx=%d aty=%d" % (tox,toy,atx,aty)
            try:
                pixelArrayVisioneCone[tox-1,toy-1] = pixelArrayDisplay[atx,aty]
            except IndexError:
                print "Index error : tox=%d toy=%d atx=%d aty=%d" % (tox,toy,atx,aty)
                pass

            # adapt the radians step to how far we are from the center (optim)
            if i>0 :
                j+=0.5/i
                #j+=0.005
            else:
                j+=0.1

            #j+=0.005
            l+=1

    #print "debug : nombre d'iterations :",l
    return visionCone

def main():
    """this function is called when the program starts.
     it initializes everything it needs, then runs in
     a loop until the function returns."""

    # Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode((1380,768))
    pygame.display.set_caption('Car racer')
    pygame.mouse.set_visible(0)

    # Create the course image
    course = load_image("course3.png")[0]
    startPosition = (100,100)
    startDirection = 0

    # Display The course
    screen.blit(course, (0, 0))
    pygame.display.flip()

    # Prepare the car Objects
    clock = pygame.time.Clock()
    car = Car(startPosition[0],startPosition[1],startDirection,2,2)
    carSprite = pygame.sprite.RenderPlain(car)

    # Prepare the cone objects
    visionConeAngle = 70
    visionConeLength = 100
    visionCone = make_vision_cone(visionConeAngle,visionConeLength)

    # Prepare input setup
    pygame.key.set_repeat(10,10)
    pause = True
    timePause = 0

    # Prepare output text
    font = pygame.font.SysFont("Arial", 15)
    helpbox = create_help_box()

    # the dataset we create
    datasetX = []
    datasetY = []

    # the neural network
    useNeuralNetwork = False
    createNeuralNetwork = False

    def buildNeuralNetwork():
        print "I have captured a dataset (X : %d elements) (Y : %d elements)" % (len(datasetX), len(datasetY))
        print "     X has %d features" % len(datasetX[0])
        print "     Y has 1 feature"
        # build the Neural Network
        startTime = time.clock()
        sys.stdout.write('Building the model....')
        sys.stdout.flush()
        # <the magic part>
        scaler = StandardScaler()
        scaler.fit(datasetX)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=1)
        clf.fit(scaler.transform(datasetX), datasetY)
        # </the magic part>
        endTime = time.clock()
        print "..done (took %d seconds)" % (endTime - startTime)
        return clf,scaler

    def updateDisplay():
        # Draw the course
        screen.fill((180, 180, 180))
        screen.blit(course, (0, 0))

        # help section
        screen.blit(helpbox,(screen.get_width()-helpbox.get_width(),screen.get_height()-helpbox.get_height()))

        # print info text
        text = ("fps=%d x=%d y=%d" % (clock.get_fps(), car.posx, car.posy))

        modeText = ""
        if useNeuralNetwork:
            modeText = "using neural network"
        else:
            modeText = "human driver"

        if pause:
            modeText += " / paused"
        elif not useNeuralNetwork:
            modeText += (" / recording (%d samples recorded)" % len(datasetX))

        if createNeuralNetwork:
            modeText = "generating model... please wait"

        screen.blit(font.render(text, 1, (0, 0, 0)), (1050, 150))
        screen.blit(font.render(modeText, 1, (0, 0, 0)), (1050, 200))

        # draw car
        carSprite.draw(screen)

        # calculate vision cone
        cone = pygame.transform.rotate(visionCone, -car.direction)
        # find the red spot in the cone (=the bottom point)
        redspot = (0, 0)
        for i in range(0, cone.get_width(), 1):
            for j in range(0, cone.get_height(), 1):
                if cone.get_at((i, j)) == Color(255, 0, 0):
                    redspot = (i, j)
                    break

        # capture what is seen by the car
        seenCone = get_vision_cone(screen, car.posx, car.posy, car.direction, visionConeAngle, visionConeLength)
        #seenCone = get_vision_cone2(screen, car.posx, car.posy, car.direction, visionConeAngle, visionConeLength)
        conelist = list(bytearray(seenCone.get_buffer().raw))

        # show what is seen by the car
        screen.blit(seenCone, (1050, 00))
        screen.blit(cone, (car.posx - redspot[0], car.posy - redspot[1]))

        # show the display
        pygame.display.flip()

        return conelist

    #Main Loop
    while True:
        # update the display
        conelist = updateDisplay()

        #Handle Input Events and generate actions
        # default action is NONE
        action = "NONE"

        events = pygame.event.get()
        for event in events:
            if event.type == QUIT:
                    return
            if event.type == KEYDOWN:
                # exit
                if event.key == K_ESCAPE:
                    return
                # pause the game
                elif event.key == K_SPACE:
                    if (time.clock() - timePause) > 1:
                        timePause = time.clock()
                        pause = not pause
                        clock.tick(60)
                # generate the neural network and use it
                elif event.key == K_g:
                    if not createNeuralNetwork and not useNeuralNetwork:
                        createNeuralNetwork = True
                        updateDisplay()
                        clf, scaler = buildNeuralNetwork()
                        createNeuralNetwork = False
                        useNeuralNetwork = True
                        car.posx = startPosition[0]
                        car.posy = startPosition[1]
                        car.direction = startDirection
                        # we need to reset the ticks
                        clock.tick(60)
                # reset to the start
                elif event.key == K_r:
                    useNeuralNetwork = False
                    car.posx = startPosition[0]
                    car.posy = startPosition[1]
                    car.direction = startDirection
                    carSprite.update(0)
                    datasetX = list()
                    datasetY = list()
                    updateDisplay()
                    pause=True
                # keys for left or right
                elif event.key == K_RIGHT:
                    action = "RIGHT"
                elif event.key == K_LEFT:
                    action = "LEFT"

        if pause:
            continue

        if useNeuralNetwork:
            # use the neural network
            datapredict=list()
            datapredict.append(conelist)
            datapredict = scaler.transform(datapredict)
            predict = clf.predict(datapredict)
            if predict == 0:
                action = "LEFT"
            elif predict == 1:
                action = "NONE"
            elif predict == 2:
                action = "RIGHT"

        if not useNeuralNetwork: datasetX.append(conelist)

        if action == "NONE":
            if not useNeuralNetwork: datasetY.append(1)
        elif action == "RIGHT":
            car.updateDirection(3)
            if not useNeuralNetwork: datasetY.append(2)
        elif action == "LEFT":
            car.updateDirection(-3)
            if not useNeuralNetwork: datasetY.append(0)

        # update the sprites
        timespent = clock.tick(30)

        if not useNeuralNetwork:
            update = max(1, timespent/30)
        else:
            update = 1

        update = 1

        carSprite.update(update)

#this calls the 'main' function when this script is executed
if __name__ == '__main__': main()