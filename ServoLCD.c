#include <stdlib.h>
#include <stdio.h>
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <pcf8574.h>
#include <lcd.h>
#include <softPwm.h>

//LCD Setup
int pcf8574_address = 0x27;
#define BASE 64
#define RS  BASE+0
#define RW  BASE+1
#define EN  BASE+2
#define LED BASE+3
#define D4  BASE+4
#define D5  BASE+5
#define D6  BASE+6
#define D7  BASE+7
int lcdhd;

//Servo Setup
#define OFFSET_MS 3
#define SERVO_MIN_MS 5+OFFSET_MS
#define SERVO_MAX_MS 25+OFFSET_MS
#define servoPin 1

//Counters for LCD Screen
int goodCount = 0;
int badCount = 0;

long map(long value,long fromLow,long fromHigh,long toLow,long toHigh){
    return (toHigh-toLow)*(value-fromLow) / (fromHigh-fromLow) + toLow;
}

void servoInit(int pin){
    softPwmCreate(pin, 0, 200);
}

void servoWrite(int pin, int angle){
    if(angle > 180) angle = 180;
    if(angle < 0) angle = 0;
    softPwmWrite(pin,map(angle,0,180,SERVO_MIN_MS,SERVO_MAX_MS));
}

//LCD Display

void updateDisplay(){
    lcdPosition(lcdhd,0,0);
    lcdPrintf(lcdhd,"Good:%d",goodCount);
    lcdPosition(lcdhd,0,1);
    lcdPrintf(lcdhd,"Bad:%d ",badCount);
}

//Main
int main(void){
    printf("Program starting...\n");
    wiringPiSetup();
    // LCD Setup
    pcf8574Setup(BASE,pcf8574_address);
    for(int i=0;i<8;i++){
        pinMode(BASE+i,OUTPUT);
    }

    digitalWrite(LED,HIGH);
    digitalWrite(RW,LOW);
    lcdhd = lcdInit(2,16,4,RS,EN,D4,D5,D6,D7,0,0,0,0);
    if(lcdhd == -1){
        printf("LCD init failed\n");
        return 1;
    }

    //Servo Setup
    servoInit(servoPin);
    updateDisplay();
    while(1){
        //SIMULATE BAD ORANGE
        //TRIGGER WITH OPENCV LATER

        servoWrite(servoPin,120);   //push orange
        delay(500);
        badCount++;                 //increment bad counter
        servoWrite(servoPin,0);     //reset servo arm
        delay(1000);
        updateDisplay();
        delay(3000);
    }
    return 0;
}
