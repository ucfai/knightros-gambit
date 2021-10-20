## Raspberry pi API code
This is the code that will run on a raspberry pi to act as an API for the higher-level code to communicate with the ESP32.

The raspberry pi sends a series of bytes via UART serial communication, which tells the ESP32 how to work. The first of which is an instruction byte. The instruction byte tells the ESP32 what the next bytes values will represent and how to use them. 

> Note: The full set of instructions to use is to be decided

### The API code manages:
* Sending bytes over UART serial communication
* Codes for different instruction commands
> Extra features are to be decided 