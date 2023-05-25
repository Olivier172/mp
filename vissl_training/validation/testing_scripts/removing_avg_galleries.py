from pathlib import Path

def main():
    p = Path("data")
    files = p.glob("*/embedding_gallery_avg*")
    for file in files:
        print("removeing" + str(file))
        file.unlink(missing_ok=True)

        
if __name__ == "__main__":
    print("Lets remove those faulty libs")
    main()