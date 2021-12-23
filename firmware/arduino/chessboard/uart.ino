char incomingByte;
char buffer[6];
int byteNum = -1; // -1 indicates that the start code hasn't been received
char currentState = '0';
char errorCode;

enum ArduinoState
{
    IDLE = '0',
    IN_PROGRESS = '1',
    EXECUTING = '2',
    END_TURN = '3',
    ERROR = '4'
};

enum MoveCommandType
{
    DIRECT = '0',
    EDGES = '1',
    ALIGN = '2'
};

enum ErrorCode
{
    NONE = '0',
    INVALID_OP = '1',
    INVALID_LOCATION = '2'
};


// Wait for input
void serialEvent2()
{
    // Loop through all available bytes
    while (Serial2.available())
    {
        // Get byte
        incomingByte = (char) Serial2.read();
        
        // Arduino is receiving a message from the Pi
        currentState = IN_PROGRESS;

        // Reset buffer position
        if (incomingByte  ==  '~')
        {
            byteNum = 0;
        }
        // Add byte to buffer
        else if (byteNum  !=  -1)
        {
            buffer[byteNum++] = incomingByte;
        }

        // Check if the buffer is full, process input
        if (byteNum  ==  6)
        {
            // Reset buffer position
            byteNum = -1;

            // Process input
            // Returns true for valid input and false for invalid input. Calls movement function
            currentState = EXECUTING;
            if (parseMessageFromPi(buffer))
                errorCode = NONE;

            // Send response
            sendMessageToPi(currentState, buffer[5], errorCode);
        }
    }
}

bool parseMessageFromPi(char * message)
{
    // Move types 0 and 1 are valid for the same range of values
    if (message[0]  ==  DIRECT  ||  message[0]  ==  EDGES)
    {
        // Check that the inputs are in a valid range
        if (message[1] < 'A'  ||  message[1] > 'L' 
           ||  message[2] < 'A'  ||  message[2] > 'L' 
           ||  message[3] < 'A'  ||  message[3] > 'L' 
           ||  message[4] < 'A'  ||   message[4] > 'L') 
        {
            errorCode = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }

        // Move type 0
        if (message[0]  ==  DIRECT)
            moveDirect(message[1] - 'A', message[2] - 'A', message[3] - 'A', message[4] - 'A');
        // Move type 1
        else if (message[0] == EDGES)
            moveAlongEdges(message[1] - 'A', message[2] - 'A', message[3] - 'A', message[4] - 'A');

    }
    else if (message[0] == ALIGN)
    {
        // Check that the inputs are in a valid range
        if (message[1] < 'A'  ||  message[1] > 'L' 
           ||  message[2] < 'A'  ||  message[2] > 'L') 
        {
            errorCode = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }

        // Center piece moved by player
        alignPiece(message[1] - 'A', message[2] - 'A');
    }
    else
    {
        // Invalid opcode
        errorCode = INVALID_OP;
        currentState = ERROR;
        return false;
    }

    // Move is valid and was made
    errorCode = NONE;
    currentState = IDLE;
    return true;
}

void sendMessageToPi(char status, char moveCount, char errorMessage)
{
    Serial2.write('~');
    Serial2.write(status);
    Serial2.write(moveCount);
    Serial2.write(errorMessage);
}
