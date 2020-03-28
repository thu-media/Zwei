# Zwei

View technical paper here. [PDF](Zwei-technical.pdf).

Video transmission algorithms should be designed to transmit video streams by balancing several contradicted metrics on demand. However, existing techniques have largely used a goal that linearly combines several weighted metrics, who often restricted mutually, which might eventually generalize a strategy that violates the original demand.

Zwei is self-play reinforcement learning algorithm that aims to tackle the video transmission tasks, which can prefectly train a policy without reward engineering.

## How to 

It's a stable version, which has already prepared the training set and the test set, and you can run the repo easily: you can jump into each folder, and just type

```
python train.py
```

instead.

We evaluate Zwei in various scenarios, not only video transmission but also general gym environment.

## Mountain Car

Here's a simple example for Zwei. Consider, we aim to teach the agent to learn the mountain car environment.
Intuitively, state-of-the-art reinforcement learning algorithm PPO can tackle this challenge in a few steps, since the goal is to maximize the overall reward.

Now let's change the challenge to: **1) touch the left margin, and 2) further, achieve the original goal.** We can see that traditional reinforcement learning fails to achieve it.
In contrast, Zwei can perfectly solve the problem, see the figure below.

<p align="center">
    <img src="demo/ppo.gif" width="40%"><img src="demo/zwei.gif" width="40%">
</p>

## Cartpole

In this scenario, we prove the simplest Zwei can solve the traditional reward-based RL problem.


## ABR (Adaptive Bit Rate Algorithm)

In this scenario, users often adopts a video player to watch the video on demand. First, video contents are pre-encoded and pre-chunked as several bitrate ladders on the server. Then the video player, placed on the client side, dynamically picks the proper bitrate for the next chunk to varying network conditions. Specifically, the bitrate decisions should achieve high bitrate and low rebuffering on the entire session. We called it adaptive bitrate streaming (ABR). In other words, Zwei should generate a policy which can obtain higher video bitrate with lower rebuffering.

We provide two types of ABR scenarios, i.e., ABR-4K and ABR-HD. Notice that Zwei uses a Rule in the two ABR tasks.

## CLS (Crowd-sourced Live Streaming)

Consider, if we were the content provider and currently we had multiple content delivery networks (CDNs) with different costs and performance, how
to schedule the users’ requests to the proper CDN, aiming to provide live streaming services withing less stalling ratio and lower cost? In common, we call that crowd-sourced live streaming(CLS).

### Experimental results

We plot the CLS's training process below. The point who plotted on the down-left corner represents better policy.

<p align="center">
    <img src="demo/lts.gif" width="50%">
</p>

## RTC (Real-time Communication)

Besides, in our daily life, we usually chat with other users instantly via a video call namely Real-Time video Communication (RTC). The RTC system consists of a sender and a receiver. The sender deploys a UDP socket channel and sends the encoded video packets to the receiver. The receiver then feeds the messages back to the
sender through feedback channel. During the session, the sender adjusts the sending bitrate for next time period, aiming
to achieve high bitrate and low round-trip time (RTT) as well as less loss ratio.

<p align="center">
    <img src="all/abw-12314-4G2.png" width="23%"><img src="all/tcp-12314-4G2.png" width="23%"><img src="all/webrtc-12314-4G2.png" width="23%"><img src="all/zwei-12314-4G2.png" width="23%">
    <img src="all/abw-12314-wifi2.png" width="23%"><img src="all/tcp-12314-wifi2.png" width="23%"><img src="all/webrtc-12314-wifi2.png" width="23%"><img src="all/zwei-12314-wifi2.png" width="23%">
    <img src="all/abw-12315-wifi3.png" width="23%"><img src="all/tcp-12315-wifi3.png" width="23%"><img src="all/webrtc-12315-wifi3.png" width="23%"><img src="all/zwei-12315-wifi3.png" width="23%">
    <img src="all/abw-12318-wifi5.png" width="23%"><img src="all/tcp-12318-wifi5.png" width="23%"><img src="all/webrtc-12318-wifi5.png" width="23%"><img src="all/zwei-12318-wifi5.png" width="23%">
    <img src="all/abw-12320-wifi5.png" width="23%"><img src="all/tcp-12320-wifi5.png" width="23%"><img src="all/webrtc-12320-wifi5.png" width="23%"><img src="all/zwei-12320-wifi5.png" width="23%">
    <img src="all/abw-12338-wifi6.png" width="23%"><img src="all/tcp-12338-wifi6.png" width="23%"><img src="all/webrtc-12338-wifi6.png" width="23%"><img src="all/zwei-12338-wifi6.png" width="23%">
    <img src="all/abw-norway-bus-1.png" width="23%"><img src="all/tcp-norway-bus-1.png" width="23%"><img src="all/webrtc-norway-bus-1.png" width="23%"><img src="all/zwei-norway-bus-1.png" width="23%">
    <img src="all/abw-norway-bus-13.png" width="23%"><img src="all/tcp-norway-bus-13.png" width="23%"><img src="all/webrtc-norway-bus-13.png" width="23%"><img src="all/zwei-norway-bus-13.png" width="23%">
    <img src="all/abw-norway-bus-14.png" width="23%"><img src="all/tcp-norway-bus-14.png" width="23%"><img src="all/webrtc-norway-bus-14.png" width="23%"><img src="all/zwei-norway-bus-14.png" width="23%">
    <img src="all/abw-norway-bus-2.png" width="23%"><img src="all/tcp-norway-bus-2.png" width="23%"><img src="all/webrtc-norway-bus-2.png" width="23%"><img src="all/zwei-norway-bus-2.png" width="23%">
    <img src="all/abw-norway-car-1.png" width="23%"><img src="all/tcp-norway-car-1.png" width="23%"><img src="all/webrtc-norway-car-1.png" width="23%"><img src="all/zwei-norway-car-1.png" width="23%">
    <img src="all/abw-norway-car-2.png" width="23%"><img src="all/tcp-norway-car-2.png" width="23%"><img src="all/webrtc-norway-car-2.png" width="23%"><img src="all/zwei-norway-car-2.png" width="23%">
    <img src="all/abw-norway-ferry-11.png" width="23%"><img src="all/tcp-norway-ferry-11.png" width="23%"><img src="all/webrtc-norway-ferry-11.png" width="23%"><img src="all/zwei-norway-ferry-11.png" width="23%">
    <img src="all/abw-norway-metro-1.png" width="23%"><img src="all/tcp-norway-metro-1.png" width="23%"><img src="all/webrtc-norway-metro-1.png" width="23%"><img src="all/zwei-norway-metro-1.png" width="23%">
    <img src="all/abw-norway-metro-10.png" width="23%"><img src="all/tcp-norway-metro-10.png" width="23%"><img src="all/webrtc-norway-metro-10.png" width="23%"><img src="all/zwei-norway-metro-10.png" width="23%">
    <img src="all/abw-norway-metro-2.png" width="23%"><img src="all/tcp-norway-metro-2.png" width="23%"><img src="all/webrtc-norway-metro-2.png" width="23%"><img src="all/zwei-norway-metro-2.png" width="23%">
    <img src="all/abw-norway-train-1.png" width="23%"><img src="all/tcp-norway-train-1.png" width="23%"><img src="all/webrtc-norway-train-1.png" width="23%"><img src="all/zwei-norway-train-1.png" width="23%">
    <img src="all/abw-norway-train-2.png" width="23%"><img src="all/tcp-norway-train-2.png" width="23%"><img src="all/webrtc-norway-train-2.png" width="23%"><img src="all/zwei-norway-train-2.png" width="23%">
    <img src="all/abw-norway-train-3.png" width="23%"><img src="all/tcp-norway-train-3.png" width="23%"><img src="all/webrtc-norway-train-3.png" width="23%"><img src="all/zwei-norway-train-3.png" width="23%">
    <img src="all/abw-norway-tram-1.png" width="23%"><img src="all/tcp-norway-tram-1.png" width="23%"><img src="all/webrtc-norway-tram-1.png" width="23%"><img src="all/zwei-norway-tram-1.png" width="23%">
    <img src="all/abw-norway-tram-2.png" width="23%"><img src="all/tcp-norway-tram-2.png" width="23%"><img src="all/webrtc-norway-tram-2.png" width="23%"><img src="all/zwei-norway-tram-2.png" width="23%">
    <img src="all/abw-norway-tram-3.png" width="23%"><img src="all/tcp-norway-tram-3.png" width="23%"><img src="all/webrtc-norway-tram-3.png" width="23%"><img src="all/zwei-norway-tram-3.png" width="23%">
    <img src="all/abw-test-0.png" width="23%"><img src="all/tcp-test-0.png" width="23%"><img src="all/webrtc-test-0.png" width="23%"><img src="all/zwei-test-0.png" width="23%">
    <img src="all/abw-test-1.png" width="23%"><img src="all/tcp-test-1.png" width="23%"><img src="all/webrtc-test-1.png" width="23%"><img src="all/zwei-test-1.png" width="23%">
    <img src="all/abw-trace-0.png" width="23%"><img src="all/tcp-trace-0.png" width="23%"><img src="all/webrtc-trace-0.png" width="23%"><img src="all/zwei-trace-0.png" width="23%">
    <img src="all/abw-trace-1.png" width="23%"><img src="all/tcp-trace-1.png" width="23%"><img src="all/webrtc-trace-1.png" width="23%"><img src="all/zwei-trace-1.png" width="23%">
    <img src="all/abw-trace-10.png" width="23%"><img src="all/tcp-trace-10.png" width="23%"><img src="all/webrtc-trace-10.png" width="23%"><img src="all/zwei-trace-10.png" width="23%">
    <img src="all/abw-trace-11.png" width="23%"><img src="all/tcp-trace-11.png" width="23%"><img src="all/webrtc-trace-11.png" width="23%"><img src="all/zwei-trace-11.png" width="23%">
    <img src="all/abw-trace-12.png" width="23%"><img src="all/tcp-trace-12.png" width="23%"><img src="all/webrtc-trace-12.png" width="23%"><img src="all/zwei-trace-12.png" width="23%">
    <img src="all/abw-trace-13.png" width="23%"><img src="all/tcp-trace-13.png" width="23%"><img src="all/webrtc-trace-13.png" width="23%"><img src="all/zwei-trace-13.png" width="23%">
    <img src="all/abw-trace-14.png" width="23%"><img src="all/tcp-trace-14.png" width="23%"><img src="all/webrtc-trace-14.png" width="23%"><img src="all/zwei-trace-14.png" width="23%">
    <img src="all/abw-trace-2.png" width="23%"><img src="all/tcp-trace-2.png" width="23%"><img src="all/webrtc-trace-2.png" width="23%"><img src="all/zwei-trace-2.png" width="23%">
    <img src="all/abw-trace-3.png" width="23%"><img src="all/tcp-trace-3.png" width="23%"><img src="all/webrtc-trace-3.png" width="23%"><img src="all/zwei-trace-3.png" width="23%">
    <img src="all/abw-trace-4.png" width="23%"><img src="all/tcp-trace-4.png" width="23%"><img src="all/webrtc-trace-4.png" width="23%"><img src="all/zwei-trace-4.png" width="23%">
    <img src="all/abw-trace-5.png" width="23%"><img src="all/tcp-trace-5.png" width="23%"><img src="all/webrtc-trace-5.png" width="23%"><img src="all/zwei-trace-5.png" width="23%">
    <img src="all/abw-trace-6.png" width="23%"><img src="all/tcp-trace-6.png" width="23%"><img src="all/webrtc-trace-6.png" width="23%"><img src="all/zwei-trace-6.png" width="23%">
    <img src="all/abw-trace-7.png" width="23%"><img src="all/tcp-trace-7.png" width="23%"><img src="all/webrtc-trace-7.png" width="23%"><img src="all/zwei-trace-7.png" width="23%">
    <img src="all/abw-trace-8.png" width="23%"><img src="all/tcp-trace-8.png" width="23%"><img src="all/webrtc-trace-8.png" width="23%"><img src="all/zwei-trace-8.png" width="23%">
    <img src="all/abw-trace-9.png" width="23%"><img src="all/tcp-trace-9.png" width="23%"><img src="all/webrtc-trace-9.png" width="23%"><img src="all/zwei-trace-9.png" width="23%">
</p>
