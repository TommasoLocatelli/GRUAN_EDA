"""
This script demonstrates how to use the Downloagpanager class to explore and download GRUAN data products (GDP) from an FTP server.
It allows the user to navigate through directories and download files interactively.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp

dir_path=r'pub/data/gruan/processing'

while True:
    items=gp.search(dir_path)
    print('-'*50)
    print(f'Items in the directory "{dir_path}":')
    for item in items:
        print(items.index(item), item)
    try:
        choice = int(input("Enter the index of the item or -1 to go back or other to quit: "))
    except:
        print("Invalid input. Please enter a number.")
        pass

    if choice==int(-1):
        dir_path = "/".join(dir_path.split("/")[:-1])

    elif choice in [items.index(item) for item in items]:
        item=items[choice]
        print("You chose:",item)
        if item.endswith('.nc'):
            gp.download(dir_path, item)
        else:
            item = "/"+item
            dir_path+=item

    else:
        print("Bye bye.")
        break