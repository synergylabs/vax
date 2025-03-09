import os
import pyedid
import time

def check_hdmi_display():
    # Path where EDID information is typically stored on Linux systems
    edid_path = "/sys/class/drm/"

    for card in os.listdir(edid_path):
        card_path = os.path.join(edid_path, card)
        if os.path.isdir(card_path):
            edid_file = os.path.join(card_path, "edid")
            status_file = os.path.join(card_path, "status")

            if os.path.exists(edid_file) and os.path.exists(status_file):
                try:
                    # Read EDID data
                    with open(edid_file, "rb") as f:
                        edid_data = f.read()

                    edid = pyedid.parse_edid(edid_data)

                    # Read display status
                    with open(status_file, "r") as f:
                        status = f.read().strip()

                    # print(f"Display detected on {card}:")
                    # print(f"Manufacturer: {edid.manufacturer}")
                    # print(f"Model: {edid.name}")
                    # print(f"Serial Number: {edid.serial}")
                    # print(f"Status: {status}")

                    return True, (status == "connected")
                except Exception as e:
                    print(f"Error reading information from {card}: {str(e)}")

    # print("No HDMI display detected")
    return False, False


if __name__ == "__main__":
    while True:
        display_connected, display_active = check_hdmi_display()
        if display_connected:
            if display_active:
                print("The display is connected and active.")
            else:
                print("The display is connected but not active.")
        else:
            print("No display is connected.")
        time.sleep(0.5)