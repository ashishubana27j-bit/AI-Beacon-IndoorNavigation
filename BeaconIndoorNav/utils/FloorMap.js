import {
  useEffect,
  useState,
} from 'react';

import {
  Dimensions,
  Image,
  StyleSheet,
  View,
} from 'react-native';

import {
  Gesture,
  GestureDetector,
} from 'react-native-gesture-handler';

import Animated, {
  useAnimatedProps,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

import Svg, {
  Circle,
  G,
  Polygon,
  Polyline,
} from 'react-native-svg';

import { FLOOR_DATA } from '../data/floorData';

const AnimatedCircle =
  Animated.createAnimatedComponent(Circle);

const AnimatedG =
  Animated.createAnimatedComponent(G);

const AnimatedPolygon =
  Animated.createAnimatedComponent(Polygon);

export default function FloorMap({
  userPosition,
  heading,
  path,
}) {
  const scale =
    useSharedValue(1);

  const savedScale =
    useSharedValue(1);

  const translateX =
    useSharedValue(0);

  const translateY =
    useSharedValue(0);

  const savedTranslateX =
    useSharedValue(0);

  const savedTranslateY =
    useSharedValue(0);

  const userX =
    useSharedValue(0);

  const userY =
    useSharedValue(0);

  const userRot =
    useSharedValue(0);

  const [
    imageLayout,
    setImageLayout,
  ] = useState(
    null
  );

  /* ---------------- image sizing ---------------- */
  useEffect(() => {
    const {
      width,
      height,
    } =
      Dimensions.get(
        'window'
      );

    const imageRatio =
      FLOOR_DATA.width /
      FLOOR_DATA.height;

    const screenRatio =
      width / height;

    let renderWidth;
    let renderHeight;
    let offsetX;
    let offsetY;

    if (
      imageRatio >
      screenRatio
    ) {
      renderWidth =
        width;

      renderHeight =
        width /
        imageRatio;

      offsetX = 0;

      offsetY =
        (height -
          renderHeight) /
        2;
    } else {
      renderHeight =
        height;

      renderWidth =
        height *
        imageRatio;

      offsetY = 0;

      offsetX =
        (width -
          renderWidth) /
        2;
    }

    setImageLayout({
      width:
        renderWidth,
      height:
        renderHeight,
      left:
        offsetX,
      top:
        offsetY,
    });
  }, []);

  /* ---------------- user update ---------------- */
  useEffect(() => {
    if (!userPosition)
      return;

    userX.value =
      withTiming(
        userPosition.x,
        {
          duration: 10,
        }
      );

    userY.value =
      withTiming(
        userPosition.y,
        {
          duration: 10,
        }
      );
     // console.log(userPosition);
  }, [
    
    userPosition,
  ]);

  useEffect(() => {
    userRot.value =
      withTiming(
        heading || 0,
        {
          duration: 80,
        }
      );
  }, [heading]);

  /* ---------------- animated props ---------------- */
  const animatedGProps =
    useAnimatedProps(
      () => ({
        rotation:
          userRot.value,
        originX:
          userX.value,
        originY:
          userY.value,
      })
    );

  const animatedUserProps =
    useAnimatedProps(
      () => ({
        cx:
          userX.value,
        cy:
          userY.value,
      })
    );

  const animatedArrowProps =
    useAnimatedProps(
      () => ({
        points: `
          ${userX.value},${userY.value - 35}
          ${userX.value + 10},${userY.value - 8}
          ${userX.value - 10},${userY.value - 8}
        `,
      })
    );

  /* ---------------- gestures ---------------- */
  const pinchGesture =
    Gesture.Pinch()
      .onUpdate(
        e => {
          scale.value =
            savedScale.value *
            e.scale;
        }
      )
      .onEnd(
        () => {
          savedScale.value =
            scale.value;
        }
      );

  const panGesture =
    Gesture.Pan()
      .onUpdate(
        e => {
          translateX.value =
            savedTranslateX.value +
            e.translationX;

          translateY.value =
            savedTranslateY.value +
            e.translationY;
        }
      )
      .onEnd(
        () => {
          savedTranslateX.value =
            translateX.value;

          savedTranslateY.value =
            translateY.value;
        }
      );

  const mapStyle =
    useAnimatedStyle(
      () => ({
        transform: [
          {
            translateX:
              translateX.value,
          },
          {
            translateY:
              translateY.value,
          },
          {
            scale:
              scale.value,
          },
        ],
      })
    );

  if (!imageLayout)
    return null;

  return (
    <View
      style={
        styles.container
      }
    >
      <GestureDetector
        gesture={Gesture.Simultaneous(
          pinchGesture,
          panGesture
        )}
      >
        <Animated.View
          style={[
            styles.mapContainer,
            mapStyle,
          ]}
        >
          <Image
            source={require('../assets/floorMap.png')}
            resizeMode="contain"
            style={{
              position:
                'absolute',
              left:
                imageLayout.left,
              top:
                imageLayout.top,
              width:
                imageLayout.width,
              height:
                imageLayout.height,
            }}
          />

          <Svg
            style={{
              position:
                'absolute',
              left:
                imageLayout.left,
              top:
                imageLayout.top,
              width:
                imageLayout.width,
              height:
                imageLayout.height,
            }}
            viewBox={`0 0 ${FLOOR_DATA.width} ${FLOOR_DATA.height}`}
          >
           {/* navigation path */}
{//console.log("PATH =", path)
}

{path?.length > 1 && (
  <Polyline
    points={path
      .map(p => `${p.x},${p.y}`)
      .join(' ')}
    fill="none"
    stroke="rgba(0,255,115,0.8)"
    strokeWidth={12}
  />
)}

            {/* beacons */}
            {FLOOR_DATA.beacons.map(
              beacon => (
                <Circle
                  key={
                    beacon.minor
                  }
                  cx={
                    beacon.x
                  }
                  cy={
                    beacon.y
                  }
                  r={
                    10
                  }
                  fill="red"
                />
              )
            )}

            {/* user */}
            {userPosition && (
              <AnimatedG
                animatedProps={
                  animatedGProps
                }
              >
                <AnimatedCircle
                  animatedProps={
                    animatedUserProps
                  }
                  r={
                    25
                  }
                  fill="rgba(0,122,255,0.2)"
                />

                <AnimatedCircle
                  animatedProps={
                    animatedUserProps
                  }
                  r={
                    10
                  }
                  fill="blue"
                />

                <AnimatedPolygon
                  animatedProps={
                    animatedArrowProps
                  }
                  fill="blue"
                />
              </AnimatedG>
            )}
          </Svg>
        </Animated.View>
      </GestureDetector>
    </View>
  );
}

const styles =
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor:
        '#000',
    },
    mapContainer: {
      flex: 1,
    },
  });