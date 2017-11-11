import random

class Player:

    name = ''
    playerclass = ''
    weaponname = ''
    weapondamage = 0
    def __init__(self, playername, classnum):

        Player.name = playername

        if classnum == 1:
            Player.playerclass = 'Knight'
        elif classnum == 2:
            Player.playerclass = 'Fighter'
        elif classnum == 3:
            Player.playerclass = 'Wizard'
        elif classnum == 4:
            Player.playerclass = 'Thief'

        weapon = Weapon(classnum)

        Player.weaponname = weapon.getWeaponName()
        Player.weapondamage = weapon.getWeaponDamage()



class Monster:

    name = ''
    weaponname = ''
    weapondamage = 0


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

        weaponnum = random.randint(1,4)
        monsweapon = Weapon(weaponnum)
        Monster.weaponname = monsweapon.getWeaponName()
        Monster.weapondamage = monsweapon.getWeaponDamage()



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
    print("\tWeapon Damage : " + str(player.weapondamage) +"\n\n")


def Continues():

    print Player.name + " begins his journey"
    print "Suddenly he faces a monster"

    newMons = Monster()
    print "The monster is " + newMons.name
    print "The " + newMons.name + " got a " + newMons.weaponname + " with a weapon damage of " + str(newMons.weapondamage)


CreatePlayer()
Continues()


