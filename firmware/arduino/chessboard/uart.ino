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
    NO_ERROR = '0',
    INVALID_OP = '1',
    INVALID_LOCATION = '2',
    INCOMPLETE_INSTRUCTION = '3',
    MOVEMENT_ERROR = '4'
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
            // Send message to Pi if the previous instruction was incomplete
            if (currentState == IN_PROGRESS)
            {
                currentState = ERROR;
                errorCode = INCOMPLETE_INSTRUCTION;
                sendMessageToPi(currentState, buffer[5], errorCode);
            }

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
            // Returns true for valid input and false for invalid input.
            currentState = EXECUTING;
            if (validateMessageFromPi(buffer))
                errorCode = NO_ERROR;


            // Sends acknowledgement and any errors with the instruction
            sendMessageToPi(currentState, buffer[5], errorCode);

            if (errorCode == NO_ERROR)
            {
                // Makes move, returns success
                if(makeMove(buffer))
                    currentState = IDLE;

                sendMessageToPi(currentState, buffer[5], errorCode);
            }
        }
    }
}

// Check that the instruction is valid
bool validateMessageFromPi(char * message)
{
    if (message[0] == DIRECT || message[0] == EDGES)
    {
        if (isInvalidCoord(message[1]) || isInvalidCoord(message[2]) 
            || isInvalidCoord(message[3]) || isInvalidCoord(message[4]))
        {
            errorCode = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else if (message[0] == ALIGN)
    {
        if (isInvalidCoord(message[1]) || isInvalidCoord(message[2]))
        {
            errorCode = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else
    {
        // Invalid opcode
        errorCode = INVALID_OP;
        currentState = ERROR;
        return false;
    }
    errorCode = NO_ERROR;
    return true;
}

bool makeMove(char * message)
{
    // Move type 0
    if (message[0]  ==  DIRECT)
    {
        if (!moveDirect(message[1] - 'A', message[2] - 'A', message[3] - 'A', message[4] - 'A'))
        {
            currentState = ERROR;
            errorCode = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 1
    else if (message[0] == EDGES)
    {
        if(!moveAlongEdges(message[1] - 'A', message[2] - 'A', message[3] - 'A', message[4] - 'A'))
        {
            currentState = ERROR;
            errorCode = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 2
    else if (message[0] == ALIGN)
    {
        if(!alignPiece(message[1] - 'A', message[2] - 'A'))
        {
            currentState = ERROR;
            errorCode = MOVEMENT_ERROR;
            return false;
        }
    }
    else
    {
        // Invalid opcode
        errorCode = INVALID_OP;
        currentState = ERROR;
        return false;
    }

    // Move is valid and was made
    errorCode = NO_ERROR;
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

bool isInvalidCoord (char c)
{
    return c < 'A' || c > 'L';
}