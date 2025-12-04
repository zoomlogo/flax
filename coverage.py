# check how many things are implemented / documented
import yaml
from colorama import Fore

from flax.encoding import codepage
from flax.builtins import transpiled_atoms, train_separators, atoms, quicks

# set of all implemented atoms/quicks/train separators
implemented = (
    transpiled_atoms.keys() | train_separators.keys() | atoms.keys() | quicks.keys()
)

# set of all documented atoms/quicks/train separators
elements_yaml = yaml.load(open("docs/elements.yaml").read(), Loader=yaml.Loader)
documented = set()
for element in elements_yaml:
    documented.add(element["element"])

# construct the set of all possible commands
codepage = set(codepage)
diagraph_starts = set("ØÆæŒœΔ")
excluded_diagraph_chars = train_separators.keys()

all_possible_commands = codepage - diagraph_starts
for diagraph_start in diagraph_starts:
    all_possible_commands |= {
        diagraph_start + char for char in codepage - train_separators.keys()
    }

# print ones which are not implmented
print(Fore.BLUE + "Not Implmented / Empty slots." + Fore.RESET)
not_implemented = 0
for i in all_possible_commands:
    if i not in implemented:
        not_implemented += 1
        print(i)

# print ones which are not documented
print(Fore.RED + "Implemented but not documented." + Fore.RESET)
not_documented = 0
for i in implemented:
    if i not in documented:
        not_documented += 1
        print(i)

print(
    Fore.BLUE
    + f"Implementation Coverage: {len(all_possible_commands) - not_implemented}/{len(all_possible_commands)} ({(1 - not_implemented / len(all_possible_commands)) * 100:.4}%)"
    + Fore.RESET
)
print(
    Fore.RED
    + f"Documentation Coverage: {len(implemented) - not_documented}/{len(implemented)} ({(1 - not_documented / len(implemented)) * 100:.4}%)"
    + Fore.RESET
)
