"""
Short script for renaming .txt files into aims format (.in). Only works for these two formats.

"""

import os
# convert all '.txt' to '.in'
if __name__=='__main__':
    
    proto_path = os.path.join(os.getcwd(), 'PROTOTYPES')
    
    material_types = os.listdir(proto_path)
    
    for material_type in material_types:
        structural_classes = os.listdir( os.path.join(proto_path, material_type) )
        for structural_class in structural_classes:
            prototypical_structures = os.listdir(os.path.join( proto_path, material_type, structural_class))
            prototypical_structures = [_ for _ in prototypical_structures if _[-4:] == '.txt']
            if len(prototypical_structures) > 1:
                raise ValueError("More than two structures in prototype path. May be polluted, please check.")
            print(prototypical_structures[0])
            prototype_path = os.path.join(proto_path, material_type, structural_class, prototypical_structures[0])
            prototypical_structures_aims = prototypical_structures[0][:-4] + '.in'
            prototype_path_new = os.path.join(proto_path, material_type, structural_class, prototypical_structures_aims)
            command_for_renaming = 'mv {} {}'.format(prototype_path, prototype_path_new)
            os.system(command_for_renaming)
    