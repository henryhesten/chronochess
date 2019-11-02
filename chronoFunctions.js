var pause = true;
var pieceD = {"k":["&#9812;","&#9818;"],"q":["&#9813;","&#9819;"],"r":["&#9814;","&#9820;"],"n":["&#9816;","&#9822;"],"b":["&#9815;","&#9821;"],"p":["&#9817;","&#9823;"]};
var dic = {"tim":0};
var side = 0;
var moveType = 0;
var selectedPiece = null;
var selectedSquare = null;
var opA = null;
var brd = null;
var stA = null;
var st = null;
var selectedTime = 0;
var selectedPastMove = false;
var toPlay = 0;
var chrono_points = [0,0]
var turn = 0;

var absCol = "#aaddaa"
var relCol = "#aaccff"
var squareWhite = "-webkit-gradient(linear,0 0, 0 100%, from(#fff), to(#eee))";//"#ffffff"
var squareGrey = "-webkit-gradient(linear,0 0, 0 100%, from(#ccc), to(#eee))";//"#cccccc"
var selCol = [[
"-webkit-gradient(linear,0 0, 0 100%, from(#ddffdd), to(#c2e3c2))",
"-webkit-gradient(linear,0 0, 0 100%, from(#ddeeff), to(#ccddee))"
],[
"-webkit-gradient(linear,0 0, 0 100%, from(#c2e3c2), to(#ddffdd))",
"-webkit-gradient(linear,0 0, 0 100%, from(#ccddee), to(#ddeeff))"
]];
var moveToCol = [[
"-webkit-gradient(linear,0 0, 0 100%, from(#cceecc), to(#b0d0b0))",
"-webkit-gradient(linear,0 0, 0 100%, from(#ccddee), to(#bbccdd))"
],[
"-webkit-gradient(linear,0 0, 0 100%, from(#b0d0b0), to(#cceecc))",
"-webkit-gradient(linear,0 0, 0 100%, from(#bbccdd), to(#ccddee))"
]];

function onLoad(){
	tabStr = '<table id="chess_board" cellpadding="0" cellspacing="0">';
	for(var i=0;i<8;i++){
		tabStr += '<tr>';
		for(var j=0;j<8;j++){
			if( side == 0 ){
				i2 = 7-i
				j2 = j
			}else{
				i2 = i
				j2 = 7-j
			}
			tabStr += '<td class="square" id="cell'+j2.toString()+i2.toString()+'" onclick="selected('+j2.toString()+','+i2.toString()+')"></td>';
		}
		tabStr += '</tr>';
	}
	tabStr += '</table>';
	document.getElementById("centre").innerHTML = tabStr;
	getState()
}

function selected(x,y){
    if(pause)
    {
        return;
    }

    selectedPastMove = null;
    piece = brd[x][y];
    var own_piece;
    if (piece == null)
    {
        own_piece = false;
    }
    else
    {
        own_piece = toPlay == piece[0];
    }

    if (selectedPiece==null)
    {
        if (piece != null && own_piece)
        {
            selectedPiece = piece;
        }
        selectedSquare=null;
    }
    else
    {
        if (piece != null && own_piece)
        {
            selectedPiece = piece;
            selectedSquare=null;
        }
        else
        {
            selectedSquare = [x,y];
        }
    }
    console.log(selectedPiece)
    console.log(selectedSquare)
    
    drawSquareColours();
}

function currentPlayer()
{
    return opA.length%2;
}
function moveClicked(i)
{
    changeTime(i);

    var move = null;
    if (i<opA.length)
    {
        move = opA[i];
    }

    if (move == null)
    {
        selectedSquare = null;
        selectedPiece = null;
    }
    else
    {
        selectedPastMove = true;
        selectedPiece = st[move[0]];
        if (move[1] == 0)
        {
            selectedSquare = [move[2],move[3]];
        }
        else
        {
            var pc = st[selectedPiece]
            selectedSquare = [move[2] + selectedPiece[3],
                            move[3] + selectedPiece[4]];
        }
        moveType = move[1];
    }

    drawAll();
}
function timeTdClicked(i)
{
    changeTime(i*2 + currentPlayer());
    if (selectedPastMove)
    {
         selectedSquare = null;
         selectedPiece = null;
    }
    drawAll();
}
function changeTime(t)
{
    if (t>=stA.length)
    {
        st = stA[stA.length-1];
        brd = brdA[stA.length-1];
    }
    else
    {
        st = stA[t];
        brd = brdA[t];
    }
    selectedTime = t;
}

function drawAll()
{
    drawState(st);
    drawMoveType();
    drawMoves();
}
function drawState(st){
	clearBoard()
	for(var i=0;i<st.length;i++){
		var pc = st[i];
		if(pc[2] == 1){ // alive
			strr = pieceD[pc[1]][pc[0]];
			document.getElementById("cell"+pc[3].toString()+pc[4].toString()).innerHTML = strr;
		}
	}
	drawSquareColours();
}
function drawMoves()
{
    inTab = "";
    inTab += "<tr><td>CP:</td><td class='cp'>" + chrono_points[0] + "</td>"
    inTab += formatTaken(null);
    inTab += "<td class='cp'>" + chrono_points[1] + "</td>"
    inTab += formatTaken(null);
    if (opA!=null)
    {
        for (var i=0; i<opA.length; i++)
        {
            if (i%2==0)
            {
                inTab += "<tr>" + drawTimeTd(i)
            }
            inTab += formatMove(i,opA[i], takenIndA[i])
            if (i%2==1 && i == opA.length-1)
            {
                inTab += "</tr>"
            }
        }

        next = opA.length
        if (next%2==1)
        {
            inTab += formatMove(next,null,-1)+ "</tr>";
            next += 1
        }

        for (var i=next; i<opA.length+6; i+=2)
        {
            inTab += "<tr>" + drawTimeTd(i) + formatMove(i,null,-1) + formatMove(i+1,null,-1)+ "</tr>";
        }
    }
    document.getElementById("moveTb").innerHTML = inTab
}
function drawTimeTd(i)
{
    var row = i/2;
    return "<td class='timeTd' onclick=timeTdClicked("+row+")>"+row+"</td>";
}
function formatMove(i,moveA, takenInd)
{
    var moveStr;
    if (moveA==null)
    {
        moveStr = "----------------";
    }
    else
    {
        piece = st[moveA[0]];
        moveStr = piece[1]+" "+moveA[0]
        if (moveA[1] == 0)
        {
            moveStr += " Abs ";
        }
        else
        {
            moveStr += " Rel ";
        }
        moveStr += moveA[2]+", "+moveA[3];
    }

    var ret = "<td id='movetd"+i+"' class='movetd";
    if (i == selectedTime)
    {
        ret += " selectedTime";
    }
    if (i == turn)
    {
        ret += " currentTurn";
    }
    ret += "' onclick='moveClicked("+i+")'>"+moveStr+"</td>";

    if (takenInd != -1)
    {
        takenP = st[takenInd];
        ret += formatTaken(takenP[1]);
    }
    else
    {
        ret += formatTaken(null);
    }
    return ret
}
function formatTaken(takenStr)
{
    ret = ""
    if (takenStr == null)
    {
        ret += "<td class='taken'></td>";
    }
    else
    {
        ret += "<td class='taken taken_"+takenStr+"'>" + takenStr + "</td>"
    }
    ret += "<td class='td_spacer'></td>";
    return ret;
}
function drawMoveType(){
    boldA = ["normal","bold"];
    document.getElementById("moveAbs").style.fontWeight = boldA[1-moveType];
    document.getElementById("moveRel").style.fontWeight = boldA[moveType];
    document.getElementById("moveAbs").style.background = absCol;
    document.getElementById("moveRel").style.background = relCol;
}
function drawSquareColours(){
    for(var i=0;i<8;i++){
        for(var j=0;j<8;j++){
            col = [squareWhite,squareGrey][(i+j+1)%2]
            for(var k=0;k<2;k++){
                color = false;
                if( (selectedSquare!=null) && ((selectedSquare[0]==i)&&(selectedSquare[1]==j)) )
                {
                    col = moveToCol[(i+j+1)%2][moveType]
                }
                if( (selectedPiece!=null) && ((selectedPiece[3]==i)&&(selectedPiece[4]==j))){
                    col = selCol[(i+j+1)%2][moveType]
                }
            }
            document.getElementById("cell"+i+j).style.background =col
        }
    }
}
function clearBoard(){
	var sqs = document.getElementsByClassName("square");
	for( var i=0; i<sqs.length; i++ ){
		sqs[i].innerHTML = "";
	}
}

function getState(){
	console.log("getState");
	pause = true;
	
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
			console.log("str",this.responseText)
			var tmpDic = JSON.parse( this.responseText );
			if( tmpDic["tim"]>dic["tim"] ){
				dic = tmpDic;
			}
			console.log(tmpDic)
			update_from_dict(dic)
			drawAll();
		}
		pause = false
	};
	xhttp.open("GET", "state", true);
	xhttp.send();
}

function updateRecord(opA){
	for( var i=0; i<opA.length; i++ ){
		op = opA[i]
		if( op==null ){
			continue
		}
	}
}

function changeMoveType(to){
    if(pause)
    {
        return;
    }
    moveType = to;
    drawMoveType();
    drawSquareColours()
}

function commitMove(){
    if(pause)
    {
        return;
    }
    if (dic["brdA"] == null || dic["brdA"] == undefined || selectedPiece==null || selectedSquare==null)
    {
        return;
    }
    
    if(selectedTime%2 != toPlay)
    {
        alert("Attempting to move the wrong colour!");
        return;
    }
    // check can move colour at this time
    
    if (moveType==0)
    {
        toX = selectedSquare[0]
        toY = selectedSquare[1]
    }
    else
    {
        toX = selectedSquare[0] - selectedPiece[3]
        toY = selectedSquare[1] - selectedPiece[4]
    }
    commit = selectedTime+","+selectedPiece[5]+","+moveType+","+toX+","+toY
    
    var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
    			console.log("return str",this.responseText)
    			var tmpDic = JSON.parse( this.responseText );
    			if( tmpDic["tim"]>dic["tim"] ){
                dic = tmpDic;
    			}
    			console.log(tmpDic)
    			update_from_dict(dic);
    			drawAll();
        }
        pause = false
    };
    xhttp.open("GET", "move?"+commit, true);
    xhttp.send();
    pause = true
}

function update_from_dict(dic)
{
    stA = dic["stA"];
    st = stA[ stA.length-1 ];
    brd = dic["brdA"][dic["brdA"].length-1];
    brdA = dic["brdA"];
    opA = dic["opA"];
    toPlay = dic["toPlay"];
    chrono_points = dic["chrono_points"]
    turn = dic["turn"];
    takenIndA = dic["takenIndA"]
    selectedTime = turn;
    selectedPiece = null;
    selectedSquare = null;
}