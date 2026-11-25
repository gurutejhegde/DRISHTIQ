# DRISHTIQ

**Viewer-Aware Streaming: Personalized Quality per Megabyte**

> DrishtiQ is a proposed video-streaming concept that aims to make better use of limited internet bandwidth by prioritizing important parts of a video and adapting optional personalized content to the viewer's network conditions.

## The Problem

Video streaming depends on a stable internet connection, but bandwidth is not always available at the same speed. It can become slow or fluctuate because of network congestion, location, or the number of people using the network.

Most adaptive streaming systems respond to slower connections by reducing the bitrate of the video. This helps prevent interruptions, but it can also reduce the clarity of the entire picture. Important details—such as a person's face, text on a screen, a scoreboard, or an object in motion—may become harder to see even when some background areas could tolerate more compression.

At the same time, streaming services may add personalized elements such as subtitles, overlays, advertisements, or alternative content. These additions can require extra data. When bandwidth is limited, delivering every optional element at full detail may not be the best use of the viewer's data allowance.

The challenge is to manage video quality, personalization, and changing network conditions together instead of treating them as separate decisions.

## Our Idea

DrishtiQ explores a viewer-aware streaming approach that tries to use each megabyte more effectively.

Instead of treating every part of a video equally, the system aims to identify which visual regions are important and preserve more quality in those areas. It also considers the viewer's preferences, where that information is available, and adapts optional personalized content according to the bandwidth available.

The goal is not simply to stream at the highest resolution. The goal is to deliver the best possible viewing experience within the available data budget, while keeping playback smooth.

## How the Proposed Solution Works

DrishtiQ is based on three connected ideas:

### 1. Give More Quality to Important Video Regions

Not every part of a video is equally important to understanding or enjoying it. In a sports broadcast, a player, the ball, and the scoreboard may matter more than parts of the stadium background. In an online lesson, the instructor, code, and presentation text may be more important than the background.

The proposed system analyses the visual content to estimate which regions may deserve more attention. It then aims to preserve more detail in those regions while allowing less important areas to use stronger compression.

The intention is to use available video data where it contributes most to the viewing experience. Visual importance is an estimate; it does not mean the system can know exactly where every viewer is looking.

### 2. Prepare Available Content in Advance

Some video content is requested by many viewers, and some upcoming content may already be available before it is needed. DrishtiQ explores using nearby or intermediate storage to keep reusable video segments and fetching likely-needed segments when network capacity permits.

For live streaming, this does not mean predicting footage that has not yet been created. Newly produced live segments can be temporarily stored and shared with viewers who request the same content. Prefetching can only prepare content that is already available, and an incorrect prediction can waste data. For that reason, advance downloads should be limited and their usefulness measured.

Personalized decisions do not require keeping a completely separate copy of the whole video for every viewer. The concept is to reuse common content where possible and adapt the selection or delivery of optional elements to each viewer's needs and available bandwidth.

### 3. Make Streaming and Personalization Decisions Together

A central decision-making component considers the available network bandwidth, the video's quality options, important visual regions, and the extra data required for personalization.

When bandwidth is sufficient, the system may deliver more personalized elements or higher visual quality. When bandwidth becomes limited, it should reduce optional personalization first and preserve the main video experience as much as possible. Reducing the base video quality is treated as a later step when the network cannot support the preferred quality.

This approach aims to balance three priorities:

- **Visual quality:** preserve details that matter to the viewing experience.
- **Personalization:** provide optional content when the available bandwidth allows it.
- **Playback continuity:** avoid unnecessary waiting and interruptions.

## Example

Imagine a student attending an online programming class over a mobile connection. The video shows the instructor, a code editor, and a presentation. The connection becomes slow, and the student has also enabled regional-language subtitles.

A conventional quality reduction may make the entire video less clear, including the code and presentation text. DrishtiQ's proposed approach would try to preserve the clarity of the important learning content, while adapting optional elements such as additional overlays to the available bandwidth. If an upcoming video segment is already available, it may also be prepared in advance when network conditions permit.

The same idea can apply to sports, entertainment, news, and other video-streaming situations where some visual information is more important than other information.

## Main Objective

To explore whether a streaming system that jointly considers visual importance, viewer preferences, personalization, and network conditions can deliver better perceived video quality per megabyte than a conventional adaptive-streaming approach, without increasing playback interruptions.

The primary measure of interest is **perceived video quality per megabyte delivered**. Playback interruptions and startup delay are also important because a visually clear stream is not useful if it continually stops.

## Project Scope

DrishtiQ is currently a project concept. The proposed approach and its components still need to be implemented, tested, and compared under consistent conditions. The project does not claim to be a deployed streaming service or a production content-delivery network.

The key questions to investigate are whether prioritizing visual regions provides a measurable benefit, whether adapting personalization helps under limited bandwidth, and whether preparing reusable upcoming content improves delivery without wasting more data than it saves.

## Expected Outcome

The intended outcome is a working demonstration and a fair comparison with a conventional adaptive-streaming baseline. The comparison should use the same video content and network conditions, and should report both video quality and the actual amount of data delivered.

DrishtiQ's central principle is simple:

**Use limited bandwidth where it matters most to the viewer.**

---

**Project name:** DRISHTIQ  
**Challenge:** Viewer-Aware Streaming — Personalized Quality per Megabyte  
**TECHgium Project ID:** 14028  
**Institution:** Anjuman Institute of Technology and Management, Bhatkal  
**Team:** Gurutej Hegde, Bibi Ruqiya Nashipudi, Sajju Hanchi, Shridhar Madival
