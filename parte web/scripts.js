function myMap() {
var mapProp= {
  center:new google.maps.LatLng(46.92627747286191, 26.373184647166422),
  zoom:15,
};
let marker;
let markerCurrent;
let pos;
  if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          pos = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
		  }
    });
markerCurrent=new google.maps.Marker({
		position: pos,
		map,
		title:"Hello World!",
	});
	}
let cityCircle;
var map = new google.maps.Map(document.getElementById("googleMap"),mapProp);
map.addListener("click", (mapsMouseEvent) => {
	if(marker==null){
	marker=new google.maps.Marker({
		position: mapsMouseEvent.latLng,
		map,
		title:"Hello World!",
	});
	}
	else{
		marker.setPosition(mapsMouseEvent.latLng);
	}
	var xhr = new XMLHttpRequest();
	xhr.open("POST", 'http://127.0.0.1:5000/');
	xhr.withCredentials = false;
	xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
	xhr.mapsMouseEvent=mapsMouseEvent;
	xhr.onreadystatechange = function() { // Call a function when the state changes.
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
		data=JSON.parse(xhr.response);
		mapsMouseEvent=this.mapsMouseEvent;
		document.getElementById("area").innerHTML = parseInt(data.area);
        if (cityCircle==null){
			cityCircle=new google.maps.Circle({
			strokeColor: "#FF0000",
			strokeOpacity: 0.8,
			strokeWeight: 2,
			fillColor: "#FF0000",
			fillOpacity: 0.35,
			map,
			center: mapsMouseEvent.latLng,
			radius: parseInt(data.area),
		});
		}
		else{
			cityCircle.setCenter(mapsMouseEvent.latLng);
			cityCircle.setRadius(parseInt(data.area));
		}
	}
	}
	xhr.send("data="+mapsMouseEvent.latLng);
});
}