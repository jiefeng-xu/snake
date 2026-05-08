# Pendulum Snake Playground

A kid-friendly Python demo for showing how a **pendulum snake** (also called a
wave pendulum) works. It opens a colorful Matplotlib window where each pendulum
has a slightly different string length, so the bobs swing with different beats
and make a moving snake-like wave.

## What to tell a 9-year-old

A pendulum is like a playground swing:

- a **shorter string** swings with a quicker beat;
- a **longer string** swings with a slower beat;
- when many pendulums start together but have different lengths, their beats
  spread out and make a rainbow snake pattern.

## Environment setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the demo

After installing the dependencies, launch the animation with:

```bash
python pendulum_snake.py
```

## Kid-friendly buttons and sliders

### Presets

- **Rainbow**: the classic pendulum snake pattern.
- **Moon**: lower gravity, so everything swings in slow motion.
- **Same beat**: every string has the same length, so the snake disappears and
  all bobs swing together.

### Playback

- **Start** resumes the animation.
- **Pause** freezes the pendulums so you can talk about the shape.
- **Restart** puts every pendulum back at the starting push.

### Sliders

- **How many?** changes the number of pendulums.
- **First string** changes the shortest string length.
- **String gap** changes how much longer each next string is.
- **Push size** changes the release angle.
- **Gravity** lets you compare places like Earth and the Moon.

## Easy demo script

Try these questions together:

1. Press **Same beat**. Ask: "Why do all the dots move together?"
2. Press **Rainbow**. Ask: "Which dots are getting ahead?"
3. Move **String gap** to `0`. Ask: "Where did the snake go?"
4. Move **String gap** higher. Ask: "How does the snake come back?"
5. Press **Moon**. Ask: "Does lower gravity make the beats faster or slower?"

## Saving a preview image

If you want a still image for notes or a README screenshot, run:

```bash
python pendulum_snake.py --save-preview pendulum_snake_preview.png
```
