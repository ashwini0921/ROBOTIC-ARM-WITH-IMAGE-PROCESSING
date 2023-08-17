# ROBOTIC-ARM-WITH-IMAGE-PROCESSING
Designed robotic Arm which can separate fruits using Deep Learning

## ABSTRACT

The project involves the development of a 6-degree-of-freedom (DOF) robotic arm capable of fruit separation through image processing techniques. A classification model is designed using Deep Neural Networks (DNN) combine with Convolutional Neural Network to accurately identify and categorize fruits based on their size, shape, and color attributes. This innovative approach combines robotics and artificial intelligence to enhance fruit sorting processes, contributing to efficiency and precision in agricultural and food industries. The integration of image processing and machine learning underscores the potential for automated fruit handling, optimizing quality control and streamlining production workflows.

## ROBOT SPECIFICATION
Drive System: Electric motor(servo motor)
Programming Software: MATLAB, ARDUINO IDE
Degree of Freedom: 5
Rotational Joints: 5
Gripper: Mechanical
Speed of Movement: Adjustable(300 degree/s to 30 degree/s)

## HARDWARE COMPONENTS USED
### Robotic Arm:
A robotic arm is a sort of arm which may be mechanical, and can be programmed with
capabilities almost equivalent to an actual arm. The robotic arm can be an aggregate of a
component or a piece of a more advanced robot.

### Servo Motors
Actuator whose motion can be controlled precisely. It is a simple electric motor with a
closed loop feedback control system for specific angular rotation.

### Arduino UNO microcontroller
A microcontroller board based on the ATmega328P is a compact integrated circuit
designed to govern a specific operation in an embedded system. It includes a processor, memory
and input/output (I/O) peripherals on a single chip.

### Ultrasonic Sensors
Ultrasonic Sensors generate ultrasonic sound waves which travel with a speed of sound in
the medium. Since it is very less absorbed by surroundings, it strikes the object and reflects. The
reflected waves are sensed by ultrasonic sensors. The time interval between to and fro motion of
ultrasonic wave is used to get distance of the object. Ultrasonic sensor detect presence and coordinates of object.
Distance of object is 340 *(t/2) m where t in seconds.

### Frontech Webcam
It has 3 MP Image Resolution and USB Interface to connect it with PC and capture image of the object. It is a CMOS based camera with 640 x 480 Pixels and maximum frame rate of 30 fps.

## SOFTWARE COMPONENT USED:

### MATLAB
It contains an image processing toolbox, a deep learning toolbox which helps in making image processing more programming compatible. MATLAB is equipped with toolbox to support Arduino programming and external camera integration to the system.

### Arduino IDE
Software to test hardware component of Robotic arm using Arduino Code before implementing in MATLAB.

## MATLAB CODE FOR IMAGE CLASSIFICATION

### STARTING WITH BUILDING A MODEL:

#### SAMPLE IMAGES OF APPLES
![apple6](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/c4416433-3973-43a2-8fbe-f54f3e76bfdf)
#### SAMPLE IMAGES OF ORANGES
![images11](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/470edbea-3732-4301-88f9-31c5be4a9cd7)
#### SAMPLE IMAGES OF BANANA
![images8](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/db59c8b2-6a33-4342-bbde-bfa4862e6d4d)


















