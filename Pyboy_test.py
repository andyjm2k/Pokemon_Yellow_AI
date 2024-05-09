from pyboy import PyBoy
pyboy = PyBoy('ROMs/Pokemon_Yellow.gbc')
game_state = 'ROMS/Pokemon_Yellow.gbc.state'
file_like_object = open(game_state, "rb")
pyboy.load_state(file_like_object)
while not pyboy.tick():
    print("Number of turns in current battle = ", pyboy.get_memory_value(0xcc29))

    # PLAYER_HP_ADDRESSES = [0xD16B, 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C]
    # total_hp = sum([pyboy.get_memory_value(address) for address in PLAYER_HP_ADDRESSES])
    # print(total_hp)
    # print("Enemy HP = ", pyboy.get_memory_value(0xcfe6))
    # value = pyboy.get_memory_value(0xd162)
    # value2 = pyboy.get_memory_value(0xcfe6)
    # value3 = pyboy.get_memory_value(0xd18b)
    # value4 = pyboy.get_memory_value(0xd1b7)
    # value5 = pyboy.get_memory_value(0xcff2)
    # value6 = pyboy.get_memory_value(0xd1e3)
    # value7 = pyboy.get_memory_value(0xd20f)
    # value8 = pyboy.get_memory_value(0xd23b)
    # value9 = pyboy.get_memory_value(0xd267)
    # value10 = pyboy.get_memory_value(0xd35d)
    # value11 = pyboy.get_memory_value(0xd360)
    # value12 = pyboy.get_memory_value(0xd361)
    # value13 = pyboy.get_memory_value(0xd364)
    # value14 = pyboy.get_memory_value(0xd056)
    # value15 = pyboy.get_memory_value(0xc0ed)
    # value3 = pyboy.get_memory_value(0xd8a3)
    # print('d162 = ', value) # No. pokemon in player party
    # print('cfe6 = ', value2) # HP
    # print('d18b = ', value3) # Poke 1 level
    # print('d1b7 = ', value4) # Poke 2 level
    # print('cff2 = ', value5) # Level
    # print('d1e3 = ', value6) # Poke 3 level
    # print('d20f = ', value7) # Poke 4 level
    # print('d23b = ', value8) # Poke 5 level
    # print('d267 = ', value9)  # Poke 6 level
    # print('d35d = ', value10)  # Current Map No.
    # print('d360 = ', value11)  # Y Position
    # print('d361 = ', value12)  # X Position
    # print('d364 = ', value13) # last map exit
    # print('d056 = ', value14)
    # print('coed =', value15)
    # print('Pokemon = ', value3)
    pass
pyboy.stop()