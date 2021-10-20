## Chessboard code
This is the code that will run on an ESP32 microcontroller to manage the chessboard system.

The ESP32 will communicate with a raspberry pi via UART serial communication, which will make the higher-lever decisions.
The arduino expects a series of bytes, the first of which is an instruction byte. The instruction byte tells the program what the next bytes values will represent and how to use them. 

> Note: The full set of instructions to use is to be decided

### The ESP32 manages:
* Two stepper motors via two Sparkfun EasyDrivers
* One electromagnet connected to power using a MOSFET (PWM controlled for variable strength)
> Extra peripherals are to be decided 