import random

class Player:

    name = ''
    playerclass = ''
    weaponname = ''
    weapondamage = 0
    hitpoints = 10
    armorpoints = 0
    exppoints = 0

    def __init__(self, playername, classnum):

        Player.name = playername

        if classnum == 1:
            Player.playerclass = 'Knight'
            Player.armorpoints = 9
        elif classnum == 2:
            Player.playerclass = 'Fighter'
            Player.armorpoints = 6
        elif classnum == 3:
            Player.playerclass = 'Wizard'
            Player.armorpoints = 5
        elif classnum == 4:
            Player.playerclass = 'Thief'
            Player.armorpoints = 4

        weapon = Weapon(classnum)

        Player.weaponname = weapon.getWeaponName()
        Player.weapondamage = weapon.getWeaponDamage()



class Monster:

    name = ''
    weaponname = ''
    weapondamage = 0
    hitpoints = 0

    def __init__(self):

        monsnum = random.randint(1, 4)
        if monsnum == 1:
            Monster.name = 'Orc'
        elif monsnum == 2:
            Monster.name = 'Elf'
        elif monsnum == 3:
            Monster.name = 'Zombie'
        elif monsnum == 4:
            Monster.name = 'Vampire'

        weaponnum = random.randint(1, 4)
        monsweapon = Weapon(weaponnum)
        Monster.weaponname = monsweapon.getWeaponName()
        Monster.weapondamage = monsweapon.getWeaponDamage()

        hpnum = random.randint(4, 10)
        Monster.hitpoints = hpnum



class Weapon:

    name = ""
    damage = 0

    def __init__(self, weaponnum):

        if weaponnum == 1:
            Weapon.name = 'Long Sword'
            Weapon.damage = 8
        elif weaponnum == 2:
            Weapon.name = 'Sword'
            Weapon.damage = 6
        elif weaponnum == 3:
            Weapon.name = 'Staff'
            Weapon.damage = 4
        elif weaponnum == 4:
            Weapon.name = 'Dagger'
            Weapon.damage = 3

    def getWeaponName(self):

        return Weapon.name

    def getWeaponDamage(self):
        return Weapon.damage


def attack():

    playerisdead = False
    monsterisdead = False
    monstinitialhp = Monster.hitpoints

    while (not playerisdead) and (not monsterisdead):
        Playerdamage = random.randint(1, 4)
        Monster.hitpoints -= Playerdamage
        print Player.name + " hits the " + Monster.name + " with a damage of " + str(Playerdamage) + "\t\t\t" + Player.name + "'s Health : " + str(Player.hitpoints) + "\t" + Monster.name + "'s Health : " + str(Monster.hitpoints)

        if Monster.hitpoints > 0:

            Monsterdamage = random.randint(1, 4)
            Player.hitpoints -= Monsterdamage
            print Monster.name + " attacked back with a damage of " + str(Monsterdamage) + "\t\t" + Player.name + "'s Health : " + str(Player.hitpoints) + "\t" + Monster.name + "'s Health : " + str(Monster.hitpoints)

            if Player.hitpoints <= 0:
                playerisdead = True

        else:
            monsterisdead = True

    if playerisdead:
        print "\nGame over"
    elif monsterisdead:
        print "\nYou defeated the " + Monster.name
        Player.exppoints = monstinitialhp
        print "You now have " + str(Player.exppoints) + " XP."






def CreatePlayer():

    name = ""
    plyrclass = 0
    print("\n##############################\n")
    print "Enter your Character's Name\n"
    Playername = raw_input(">>")

    print "\nChoose the Class of your character with the given class numbers"
    print "\n(1) Knight \n(2) Fighter \n(3) Wizard \n(4) Thief\n"
    Playerclass = input()

    player = Player(Playername, Playerclass)
    print("\tPlayer Name : " + player.name )
    print("\tPlayer Class: " + player.playerclass)
    print("\tPlayer Weapon : " + player.weaponname)
    print("\tWeapon Damage : " + str(player.weapondamage))
    print "\tHealth : " + str(player.hitpoints)
    print "\tArmor : " + str(player.armorpoints)
    print "\n\n"


def Continues():

    print Player.name + " begins his journey"
    print "Suddenly he faces a monster"


    newMons = Monster()
    print "The monster is " + newMons.name
    print "The " + newMons.name + " got a " + newMons.weaponname + " with a weapon damage of " + str(newMons.weapondamage) + " and Health of " + str(newMons.hitpoints)

    decision = 0
    print "Do you want to (1) fight or (2) run ???"
    decision = input()
    if decision == 1:
        attack()
    elif decision == 2:
        escpropability = random.randint(1,4)
        if escpropability == 1:
            print "You fled successfully."
        else:
            print "The monster came for a fight. He will get a fight. Your escape plan was a failure."
            attack()
    else:
        print "Invalid input"

CreatePlayer()
Continues()


