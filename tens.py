#           Tens est un format de fichier qui permet de stocker des tenseurs en binaire.
#       Composition du fichier:
# Données en Little-Endian ('<')
# 0x0000                    : 2 bytes       : char[2],      Nombre magique : Cc en ascii
# 0x0002                    : 2 bytes       : uint16,       Type de données (voir classe DataType)
# 0x0004                    : 4 bytes       : uint,         Nombre de dimensions
# 0x0008                    : 4 bytes       : uint,         Taille de la dimension 0
# 0x0???                    : 4 bytes       : uint,         ...
# 0x0008 + 4 * n            : 4 bytes       : uint,         Taille de la dimension n
# 0x000C + 4 * n            : (Variable)    : char[],       Donnée de l'élément 0
# 0x0???                    : (Variable)    : char[],       ...
# 0x0008 + 4 * n + s * e    : (Variable)    : char[],       Donnée de l'élément de fin 


import struct

import numpy as np



class DataType():
    """ Types de données présentes dans le fichier (sous forme d'enum) """
    # (ID, dtype, conversion struct, taille en octets)
    SPECIAL         = (0, "",           "",   0)       # Type inconnu
    BYTE            = (1, "uint8",      "<B", 1)  # Equivalent à UINT8        
    INT8            = (2, "int8",       "<b", 1)        
    INT32           = (3, "int32",      "<i", 4)
    FLOAT32         = (4, "float32",    "<f", 4)
    INT64           = (5, "int64",      "<l", 8)
    FLOAT64         = (6, "float64",    "<d", 8)
    UINT32          = (7, "uint32",     "<I", 4)
    UINT64          = (8, "uint64",     "<L", 8)
    BOOL            = (9, "bool",       "<?", 1)

    # Tous les types sauf SPECIAL
    ALL = [
        BYTE,
        INT8,
        INT32,
        FLOAT32,
        INT64,
        FLOAT64,
        UINT32,
        UINT64,
        BOOL,
        ]


    @staticmethod
    def get_tensor_type(tensor):
        """ Renvoie le type de données du tenseur """
        # On cherche dans tous les types
        for t in DataType.ALL:
            if t[1] == tensor.dtype:
                return t

        # Erreur, renvoie SPECIAL
        return SPECIAL

    @staticmethod
    def get_data_dtype(data_type):
        """ Renvoie le type de dtype à partir du type de donnée (int) """
        # On cherche dans tous les types
        for t in DataType.ALL:
            if t[0] == data_type:
                return t

        # Erreur, renvoie SPECIAL
        return DataType.SPECIAL



def load(file_path):
    """ Charge un tenseur à partir de son emplacement """
    with open(file_path, "rb") as f:
        # Nombre magique
        magic_nb = f.read(2)
        if magic_nb != b"Cc":
            raise Exception("Fichier tens inconnu (nombre magique)")

        # Type de données
        data_type = DataType.get_data_dtype(struct.unpack("<H", f.read(2))[0])
        if data_type == DataType.SPECIAL:
            raise Exception("Type de tenseur inconnu ou non supporté")

        # Nombre de dimensions
        dimensions = struct.unpack("<I", f.read(4))[0]
        
        # Taille de chaque dimension
        shape = []
        size = 1
        for d in range(dimensions):
            s = struct.unpack("<I", f.read(4))[0]
            shape.append(s)
            size *= s


        # Données
        tensor = np.empty([size], dtype=data_type[1])
        for i in range(size):
            d = struct.unpack(data_type[2], f.read(data_type[3]))[0]
            tensor[i] = d

        tensor.shape = shape

        return tensor



def save(tensor, file_path):
    """ Enregistre le tenseur à l'emplacement file_path """
    with open(file_path, "wb") as f:
        # Nombre magique
        f.write(b"Cc")

        # Type de données
        data_type = DataType.get_tensor_type(tensor)
        if data_type == DataType.SPECIAL:
            raise Exception("Type de tenseur inconnu ou non supporté")

        f.write(struct.pack("<H", data_type[0]))

        # Nombre de dimensions
        f.write(struct.pack("<I", len(tensor.shape)))

        # Taille de chaque dimension
        for s in tensor.shape:
            f.write(struct.pack("<I", s))

        # Données
        for d in np.nditer(tensor):
            f.write(struct.pack(data_type[2], d))
