declare module "bubblesets" {
  export class BSplineShapeGenerator {
    constructor();
  }

  export class BubbleSet {
    constructor();
    createOutline(
      rectangles: Array<{
        x: number;
        y: number;
        width: number;
        height: number;
      }>,
      exclusions: Array<any>,
      connections: any
    ): any;

    static addPadding(
      rectangles: Array<{
        x: number;
        y: number;
        width: number;
        height: number;
      }>,
      padding: number
    ): Array<{ x: number; y: number; width: number; height: number }>;
  }

  export class PointPath {
    constructor(list: any);
    transform(operations: any[]): any;
  }

  export class ShapeSimplifier {
    constructor(value: number);
  }
}
