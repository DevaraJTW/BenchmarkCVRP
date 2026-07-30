def getModelFile(CVRPNumber: int, architecture: str):
    filename = ""
    filename += architecture
    filename += "/CVRP" + str(CVRPNumber)
    filename += "/lightning_logs/version_0/checkpoints/"
    
    match CVRPNumber:
        case 20:
            filename += "epoch=149-step=29400.ckpt"
        case 50:
            filename += "epoch=149-step=29400.ckpt"
        case 100:
            filename += "epoch=149-step=29400.ckpt"
    
    return filename