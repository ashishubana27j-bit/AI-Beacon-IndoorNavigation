//floordata.js
export const FLOOR_DATA = {
  width: 2000,
  height: 1500,


  beacons: [
    { minor: "4113", x: 315, y: 950, latitude:25.26417611 ,longitude:55.38526176 , name: "office" },
    { minor: "4114", x: 550, y: 1260,latitude:25.26418020 ,longitude:55.38534943, name: "turn1" },//25.26418020N, 55.38534943E
    { minor: "4115", x: 602, y: 1123,latitude:25.26420767 ,longitude:55.38535244, name: "turn2" },//25.26420767N, 55.38535244E
    { minor: "4116", x: 654, y: 1000,latitude:25.26424744 ,longitude:55.38531217, name: "entryGate" },//25.26424744N, 55.38531217E
    { minor: "4117", x: 440, y: 1120,latitude:25.26424744 ,longitude:55.38531217, name: "mid_center" }
    //{ minor: "4117", x: 435, y: 1116,latitude:25.26424744 ,longitude:55.38531217, name: "entrymid" }
  ],

  walkNodes: [
    { id:'A', x:630, y:920,  links:['B'] },
    { id:'B', x:630, y:1030, links:['A','C','D'] },
    { id:'C', x:550, y:1030, links:['B'] },
    { id:'D', x:630, y:1028, links:['E','B'] },
    { id:'E', x:630, y:1230, links:['D','F'] },
    { id:'F', x:520, y:1230, links:['E','G'] },
    { id:'G', x:430, y:1145, links:['F','H'] },
    { id:'H', x:370, y:1075, links:['G','I'] },
    { id:'I', x:340, y:1000, links:['H','J'] },
    { id:'J', x:280, y:980,  links:['I'] },
  ],
 destinations: [
    { id: 'room1', name: 'Office', nodeId: 'J' },
    { id: 'room2', name: 'Entry Gate', nodeId: 'A' },
    { id: 'room3', name: 'Washroom', nodeId: 'C' },
  ],

  fingerprints:[
   
  ],
};